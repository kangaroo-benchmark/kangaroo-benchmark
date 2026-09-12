"""Build the paired German-English subset for the language ablation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from analysis.inputs import file_sha256, load_dataset

LETTERS = "ABCDE"
SAMPLE_COLUMNS = [
    "id",
    "year",
    "group",
    "points",
    "problem_number",
    "multimodal",
    "visual_type",
    "problem_statement",
    *[f"sol_{letter}" for letter in LETTERS],
    "answer",
]
TRANSLATED_FIELDS = [
    "problem_statement",
    *[f"sol_{letter}" for letter in LETTERS],
]
EXCLUDED_ITEMS = {
    "02_910_p1_2b36672f-c058-4e09-aa05-7ba860ad0fce": (
        "required diagram not extracted separately from the German question crop"
    ),
    "98_34_p2_e9e47552-ed27-4004-8c53-6dd68f8cab74": "German text in diagram",
    "02_56_p2_428d4cb9-65a8-4139-b4ce-9fcb52611ea3": "German text in diagram",
    "08_56_p2_83b558ad-0ce7-4531-b8ee-03f6f66568d9": "German text in answer images",
    "12_1113_p1_a41f148f-3299-4208-8d37-00a27aa21f1d": "German text in diagram",
    "15_910_p3_55c56e37-fc71-4192-813c-983fd3b0ce97": "German text in diagram",
    "15_56_p2_befcf006-a986-4810-b05e-6f0714ace9e6": "German text in diagram",
    "21_78_p1_0fb36dde-8a4d-43b1-85ad-bfde1222082f": ("German text in answer images"),
    "24_78_p2_58c19a9f-c23d-438a-8cc5-61096d543e0b": "German text in diagram",
}


# Author review of the translated subset before evaluation; the archived translation
# file fixes the resulting 200 items.
POST_REVIEW = {
    "corrections": {
        "04_34_p3_42bbe9bb-5bf2-43ce-933f-4a8d97f619da": (
            "option E revised to 'It depends on the number.' for register consistency"
        ),
        "18_56_p2_0574b4b2-f7f0-4a0e-8b66-6782597e45a9": (
            "statement revised to 'a triangle with three sides of equal length' to "
            "preserve the source's non-technical phrasing"
        ),
    },
    "exclusions": {
        "08_56_p3_c55dce89-2002-42c5-8bc7-3b61a271a304": (
            "German source text is OCR-corrupted (question fragments interleaved into "
            "options B and D) while the English arm was silently repaired; pair "
            "excluded to preserve arm symmetry"
        ),
    },
    "replacements": {
        "08_56_p3_c55dce89-2002-42c5-8bc7-3b61a271a304": {
            "replaced_by": "08_56_p3_715e75a9-21dc-458f-9698-824cb6ceb8bb",
            "selection_rule": (
                "same year-group-points-visual_type cell (2008, 5-6, 4, "
                "question_diagram_only); lowest SHA-256 rank under the original seed "
                "objective among unsampled candidates passing the exclusion criteria"
            ),
            "translation": "manual author-reviewed translation added to the translation file",
        },
    },
}


def add_visual_type(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    has_diagram = frame["associated_images_bin"].map(len) > 0
    image_columns = [f"sol_{letter}_image_bin" for letter in LETTERS]
    has_image_answers = frame[image_columns].notna().any(axis=1)
    frame["visual_type"] = np.select(
        [
            has_diagram & has_image_answers,
            has_diagram,
            has_image_answers,
        ],
        [
            "question_diagram_and_image_answers",
            "question_diagram_only",
            "image_answers_only",
        ],
        default="no_separate_visual_element",
    )
    return frame


def allocate_proportional_quotas(
    dataset: pd.DataFrame, strata: list[str], sample_size: int
) -> pd.DataFrame:
    counts = dataset.groupby(strata, observed=True).size().rename("population_count")
    quotas = counts.reset_index()
    quotas["exact_quota"] = quotas["population_count"] * sample_size / len(dataset)
    quotas["sample_quota"] = np.floor(quotas["exact_quota"]).astype(int)
    remaining = sample_size - int(quotas["sample_quota"].sum())
    quotas["remainder"] = quotas["exact_quota"] - quotas["sample_quota"]
    order = quotas.sort_values(
        ["remainder", *strata], ascending=[False, *([True] * len(strata))]
    ).index
    quotas.loc[order[:remaining], "sample_quota"] += 1
    if int(quotas["sample_quota"].sum()) != sample_size:
        raise ValueError("Proportional allocation did not reach the requested size")
    if (quotas["sample_quota"] > quotas["population_count"]).any():
        raise ValueError("A sampling quota exceeds its stratum size")
    return quotas.sort_values(strata).reset_index(drop=True)


def select_translation_subset(
    dataset: pd.DataFrame,
    sample_size: int = 200,
    seed: int = 20260728,
    fixed_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the quota-balanced subset, or verify that ``fixed_ids`` (the archived
    subset) meet every quota exactly."""
    if len(dataset) != 3886 or dataset["id"].nunique() != 3886:
        raise ValueError("Expected 3,886 unique items in the dataset")
    frame = add_visual_type(dataset)
    frame = frame.loc[~frame["id"].astype(str).isin(EXCLUDED_ITEMS)].copy()
    strata = ["group", "points", "visual_type"]
    joint_quotas = allocate_proportional_quotas(frame, strata, sample_size)
    year_quotas = allocate_proportional_quotas(frame, ["year"], sample_size)

    constraint_rows = []
    targets = []
    for quota in joint_quotas.itertuples(index=False):
        mask = np.ones(len(frame), dtype=bool)
        for column in strata:
            mask &= frame[column].to_numpy() == getattr(quota, column)
        constraint_rows.append(mask.astype(float))
        targets.append(int(quota.sample_quota))
    for quota in year_quotas.itertuples(index=False):
        constraint_rows.append((frame["year"].to_numpy() == quota.year).astype(float))
        targets.append(int(quota.sample_quota))

    objective = (
        frame["id"]
        .astype(str)
        .map(
            lambda item_id: (
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()[:8],
                    "big",
                )
                / 2**64
            )
        )
    )
    target_array = np.asarray(targets, dtype=float)
    constraint_matrix = np.vstack(constraint_rows)
    if fixed_ids is None:
        result = milp(
            c=objective.to_numpy(),
            integrality=np.ones(len(frame)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                csr_matrix(constraint_matrix), target_array, target_array
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise ValueError(
                f"Could not construct the balanced sample: {result.message}"
            )
        selected = result.x > 0.5
    else:
        selected = frame["id"].astype(str).isin(fixed_ids).to_numpy()
        if selected.sum() != len(set(fixed_ids)):
            raise ValueError("Fixed sample ids are not all eligible dataset items")
    if not np.array_equal(constraint_matrix @ selected.astype(float), target_array):
        raise ValueError("The translation subset does not meet every sampling quota")
    sample = frame.loc[selected].sort_values(["year", "group", "problem_number"])
    sample = sample.reset_index(drop=True)
    if len(sample) != sample_size or sample["id"].nunique() != sample_size:
        raise ValueError("The selected translation subset is not unique and complete")
    joint_quotas.insert(0, "quota_type", "group_points_visual_type")
    year_quotas.insert(0, "quota_type", "year")
    quotas = pd.concat([joint_quotas, year_quotas], ignore_index=True, sort=False)
    return sample, quotas


def build_sampling_report(dataset: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    population = add_visual_type(dataset)
    population["year"] = pd.to_numeric(population["year"])
    population["year_band"] = pd.cut(
        population["year"],
        bins=[1997, 2004, 2011, 2018, 2025],
        labels=["1998-2004", "2005-2011", "2012-2018", "2019-2025"],
    )
    selected = sample.copy()
    selected["year"] = pd.to_numeric(selected["year"])
    selected["year_band"] = pd.cut(
        selected["year"],
        bins=[1997, 2004, 2011, 2018, 2025],
        labels=["1998-2004", "2005-2011", "2012-2018", "2019-2025"],
    )

    records = []
    for dimension in [
        "group",
        "points",
        "visual_type",
        "multimodal",
        "year_band",
        "year",
    ]:
        population_counts = population[dimension].value_counts(dropna=False)
        sample_counts = selected[dimension].value_counts(dropna=False)
        categories = sorted(
            set(population_counts.index) | set(sample_counts.index), key=str
        )
        for category in categories:
            population_count = int(population_counts.get(category, 0))
            sample_count = int(sample_counts.get(category, 0))
            population_share = population_count / len(population)
            sample_share = sample_count / len(selected)
            records.append(
                {
                    "dimension": dimension,
                    "category": str(category),
                    "population_count": population_count,
                    "population_share": population_share,
                    "sample_count": sample_count,
                    "sample_share": sample_share,
                    "difference_percentage_points": 100
                    * (sample_share - population_share),
                }
            )
    return pd.DataFrame(records)


def load_translations(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("translations")
    if not isinstance(records, list):
        raise ValueError("Translation file must contain a translations list")
    translations = pd.DataFrame(records)
    required = {"id", *[f"english_{field}" for field in TRANSLATED_FIELDS]}
    missing = required - set(translations.columns)
    if missing:
        raise ValueError(f"Translation file is missing columns: {sorted(missing)}")
    if translations["id"].duplicated().any():
        raise ValueError("Translation file contains duplicate item identifiers")
    sample_ids = set(sample["id"].astype(str))
    translation_ids = set(translations["id"].astype(str))
    if sample_ids != translation_ids:
        raise ValueError(
            "Translation identifiers do not match the selected sample: "
            f"missing={sorted(sample_ids - translation_ids)}, "
            f"extra={sorted(translation_ids - sample_ids)}"
        )

    merged = sample.merge(translations, on="id", validate="one_to_one")
    for field in TRANSLATED_FIELDS:
        english_field = f"english_{field}"
        source_present = merged[field].fillna("").astype(str).str.strip().ne("")
        translation_present = (
            merged[english_field].fillna("").astype(str).str.strip().ne("")
        )
        missing_translation = source_present & ~translation_present
        unexpected_translation = ~source_present & translation_present
        if missing_translation.any() or unexpected_translation.any():
            bad_ids = merged.loc[
                missing_translation | unexpected_translation, "id"
            ].tolist()
            raise ValueError(f"Translation coverage mismatch for {field}: {bad_ids}")
    return merged


def build_evaluation_frames(
    translated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_columns = [
        column
        for column in translated.columns
        if not column.startswith(("english_", "german_"))
    ]
    german = translated[source_columns].copy()
    english = german.copy()
    german["source_id"] = german["id"].astype(str)
    english["source_id"] = english["id"].astype(str)
    german["id"] = german["id"].astype(str) + "_de_control"
    english["id"] = english["id"].astype(str) + "_en"
    german["language"] = "de"
    english["language"] = "en"
    for field in TRANSLATED_FIELDS:
        german_field = f"german_{field}"
        if german_field in translated:
            normalized = translated[german_field].notna()
            german.loc[normalized, field] = translated.loc[normalized, german_field]
        english[field] = translated[f"english_{field}"]

    # The full question crop contains the German statement. Both paired arms omit
    # it and retain only the separately extracted diagrams and image-based options.
    german["question_image"] = None
    english["question_image"] = None
    return german, english


def image_data_url(value: object) -> str | None:
    if not isinstance(value, (bytes, bytearray)):
        return None
    mime = "image/png" if bytes(value).startswith(b"\x89PNG") else "image/jpeg"
    encoded = base64.b64encode(bytes(value)).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_review_html(translated: pd.DataFrame, output_path: Path) -> None:
    items = []
    for row in translated.itertuples(index=False):
        german = {}
        for field in TRANSLATED_FIELDS:
            normalized_field = f"german_{field}"
            normalized_value = getattr(row, normalized_field, None)
            german[field] = (
                normalized_value
                if isinstance(normalized_value, str)
                else getattr(row, field)
            )
        record = {
            "id": str(row.id),
            "year": int(row.year),
            "group": str(row.group),
            "points": int(row.points),
            "problem_number": str(row.problem_number),
            "multimodal": bool(row.multimodal),
            "visual_type": str(row.visual_type),
            "answer": str(row.answer),
            "german": german,
            "english": {
                field: getattr(row, f"english_{field}") for field in TRANSLATED_FIELDS
            },
            "question_image": image_data_url(row.question_image),
            "associated_images": [
                image_data_url(value) for value in row.associated_images_bin
            ],
            "option_images": {
                letter: image_data_url(getattr(row, f"sol_{letter}_image_bin"))
                for letter in LETTERS
            },
        }
        items.append(record)

    payload = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kangaroo translation review</title>
<style>
:root {{ color-scheme: light dark; --bg:#f4f1eb; --paper:#fffdf8; --ink:#23211f; --muted:#6f6a63; --line:#d8d1c7; --accent:#285d4b; --warn:#9b4f2f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ position:sticky; top:0; z-index:4; background:color-mix(in srgb,var(--bg) 93%,transparent); backdrop-filter:blur(12px); border-bottom:1px solid var(--line); }}
.top {{ max-width:1500px; margin:auto; padding:14px 24px 12px; }}
h1 {{ margin:0 0 10px; font:650 22px/1.2 ui-serif,Georgia,serif; letter-spacing:-.01em; }}
.intro {{ margin:-4px 0 10px; color:var(--muted); font-size:13px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
input,select,button,textarea {{ font:inherit; color:inherit; background:var(--paper); border:1px solid var(--line); border-radius:7px; }}
input,select,button {{ height:36px; padding:0 10px; }}
input[type=search] {{ min-width:250px; flex:1; }}
button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.progress {{ margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }}
main {{ max-width:1500px; margin:auto; padding:22px 24px 80px; }}
.item {{ background:var(--paper); border:1px solid var(--line); border-radius:10px; margin:0 0 18px; overflow:hidden; scroll-margin-top:104px; }}
.item.issue {{ border-color:var(--warn); }} .item.approved {{ border-color:var(--accent); }}
.item-head {{ display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }}
.item-title {{ font-weight:650; }} .meta {{ color:var(--muted); }} .item-actions {{ margin-left:auto; display:flex; gap:12px; }}
.item-actions label {{ cursor:pointer; }}
.columns {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); }}
.language {{ padding:18px; min-width:0; }} .language + .language {{ border-left:1px solid var(--line); }}
h2 {{ margin:0 0 12px; color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }}
.statement {{ white-space:pre-wrap; font-size:16px; margin-bottom:16px; overflow-wrap:anywhere; }}
.options {{ display:grid; gap:8px; }} .option {{ display:grid; grid-template-columns:28px minmax(0,1fr); gap:8px; padding:8px 0; border-top:1px solid color-mix(in srgb,var(--line) 65%,transparent); }}
.option.correct .letter {{ color:var(--accent); font-weight:750; }} .option-text {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
.visuals {{ padding:0 18px 18px; display:flex; flex-wrap:wrap; gap:12px; border-top:1px solid var(--line); }}
.visuals figure {{ margin:16px 0 0; max-width:100%; }} .visuals img {{ display:block; max-width:min(100%,740px); max-height:520px; border:1px solid var(--line); background:white; }}
figcaption {{ margin-top:5px; color:var(--muted); font-size:12px; }}
.notes {{ width:calc(100% - 36px); min-height:60px; margin:0 18px 18px; padding:9px 10px; resize:vertical; }}
.empty {{ padding:80px 20px; text-align:center; color:var(--muted); }}
@media (max-width:820px) {{ .columns {{ grid-template-columns:1fr; }} .language + .language {{ border-left:0; border-top:1px solid var(--line); }} .progress {{ width:100%; margin:0; }} }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#191816; --paper:#24221f; --ink:#f0ece5; --muted:#aaa39a; --line:#454039; --accent:#77b79e; --warn:#db8a64; }} }}
</style>
</head>
<body>
<header><div class="top">
  <h1>Kangaroo German–English translation review</h1>
  <p class="intro">Review state stays in this browser. The original crop is shown for checking but is omitted from both evaluation arms. Use [ and ] to move between pending items.</p>
  <div class="controls">
    <input id="search" type="search" placeholder="Search text or item ID" aria-label="Search">
    <select id="status"><option value="all">All statuses</option><option value="pending">Pending</option><option value="approved">Approved</option><option value="issue">Needs revision</option></select>
    <select id="group"><option value="all">All grades</option></select>
    <select id="points"><option value="all">All points</option></select>
    <select id="visual"><option value="all">All visual types</option></select>
    <button id="prev" type="button">Previous pending</button><button id="next" type="button">Next pending</button>
    <button id="export" type="button">Export review</button>
    <span class="progress" id="progress"></span>
  </div>
</div></header>
<main id="items"></main>
<script>
const items={payload};
const storageKey="kangaroo-translation-review-v1";
let review=JSON.parse(localStorage.getItem(storageKey)||"{{}}");
const esc=value=>String(value??"").replace(/[&<>\"']/g,ch=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#039;"}}[ch]));
const text=value=>esc(value).replace(/\\n/g,"<br>");
const filters={{search:"",status:"all",group:"all",points:"all",visual:"all"}};
function state(id) {{ return review[id]||{{status:"pending",notes:""}}; }}
function save() {{ localStorage.setItem(storageKey,JSON.stringify(review)); updateProgress(); }}
function setStatus(id,status) {{ review[id]={{...state(id),status}}; save(); document.getElementById(`item-${{id}}`)?.classList.toggle("approved",status==="approved"); document.getElementById(`item-${{id}}`)?.classList.toggle("issue",status==="issue"); }}
function options(item,lang) {{ return "ABCDE".split("").map(letter=>`<div class="option ${{item.answer===letter?"correct":""}}"><span class="letter">${{letter}}</span><span class="option-text">${{text(item[lang][`sol_${{letter}}`])||"<em>Image only</em>"}}</span></div>`).join(""); }}
function visuals(item) {{ const figures=[]; if(item.question_image) figures.push(`<figure><img loading="lazy" src="${{item.question_image}}" alt="Original question crop"><figcaption>Original German question crop, shown for review only</figcaption></figure>`); item.associated_images.forEach((src,i)=>{{if(src) figures.push(`<figure><img loading="lazy" src="${{src}}" alt="Diagram ${{i+1}}"><figcaption>Extracted question diagram ${{i+1}}</figcaption></figure>`)}}); for(const [letter,src] of Object.entries(item.option_images)) if(src) figures.push(`<figure><img loading="lazy" src="${{src}}" alt="Option ${{letter}} image"><figcaption>Option ${{letter}} image</figcaption></figure>`); return figures.length?`<div class="visuals">${{figures.join("")}}</div>`:""; }}
function matches(item) {{ const s=state(item.id); const hay=[item.id,item.year,item.group,item.points,item.visual_type,...Object.values(item.german),...Object.values(item.english)].join(" ").toLowerCase(); return (!filters.search||hay.includes(filters.search))&&(filters.status==="all"||s.status===filters.status)&&(filters.group==="all"||item.group===filters.group)&&(filters.points==="all"||String(item.points)===filters.points)&&(filters.visual==="all"||item.visual_type===filters.visual); }}
function render() {{ const visible=items.filter(matches); document.getElementById("items").innerHTML=visible.length?visible.map(item=>{{const s=state(item.id);return `<article class="item ${{s.status}}" id="item-${{item.id}}"><div class="item-head"><span class="item-title">${{esc(item.id)}}</span><span class="meta">${{item.year}} · grades ${{item.group}} · ${{item.points}} points · task ${{esc(item.problem_number)}} · ${{esc(item.visual_type.replaceAll("_"," "))}}</span><span class="item-actions"><label><input type="radio" name="status-${{item.id}}" ${{s.status==="approved"?"checked":""}} onchange="setStatus('${{item.id}}','approved')"> Approved</label><label><input type="radio" name="status-${{item.id}}" ${{s.status==="issue"?"checked":""}} onchange="setStatus('${{item.id}}','issue')"> Needs revision</label><label><input type="radio" name="status-${{item.id}}" ${{s.status==="pending"?"checked":""}} onchange="setStatus('${{item.id}}','pending')"> Pending</label></span></div><div class="columns"><section class="language"><h2>German source</h2><div class="statement">${{text(item.german.problem_statement)}}</div><div class="options">${{options(item,"german")}}</div></section><section class="language"><h2>English translation</h2><div class="statement">${{text(item.english.problem_statement)}}</div><div class="options">${{options(item,"english")}}</div></section></div>${{visuals(item)}}<textarea class="notes" placeholder="Review notes" aria-label="Notes for ${{item.id}}" oninput="review['${{item.id}}']={{...state('${{item.id}}'),notes:this.value}};save()">${{esc(s.notes)}}</textarea></article>`}}).join(""):"<div class='empty'>No items match these filters.</div>"; updateProgress(); }}
function updateProgress() {{ const counts={{approved:0,issue:0,pending:0}}; items.forEach(item=>counts[state(item.id).status]++); document.getElementById("progress").textContent=`${{counts.approved}} approved · ${{counts.issue}} issues · ${{counts.pending}} pending`; }}
function fillSelect(id,values) {{ const el=document.getElementById(id); values.forEach(value=>el.insertAdjacentHTML("beforeend",`<option value="${{esc(value)}}">${{esc(value.replaceAll?.("_"," ")??value)}}</option>`)); }}
fillSelect("group",[...new Set(items.map(x=>x.group))].sort()); fillSelect("points",[...new Set(items.map(x=>String(x.points)))].sort()); fillSelect("visual",[...new Set(items.map(x=>x.visual_type))].sort());
for(const id of ["status","group","points","visual"]) document.getElementById(id).addEventListener("change",e=>{{filters[id]=e.target.value;render();}}); document.getElementById("search").addEventListener("input",e=>{{filters.search=e.target.value.toLowerCase().trim();render();}});
function jump(direction) {{ const pending=items.filter(item=>state(item.id).status==="pending"&&matches(item)); if(!pending.length)return; const current=items.findIndex(item=>location.hash===`#item-${{item.id}}`); const ordered=direction>0?pending:[...pending].reverse(); const target=ordered.find(item=>direction>0?items.indexOf(item)>current:items.indexOf(item)<current)||ordered[0]; location.hash=`item-${{target.id}}`; }}
document.getElementById("prev").onclick=()=>jump(-1); document.getElementById("next").onclick=()=>jump(1); document.getElementById("export").onclick=()=>{{const blob=new Blob([JSON.stringify({{exported_at:new Date().toISOString(),review}},null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="translation_review.json";a.click();URL.revokeObjectURL(a.href);}};
document.addEventListener("keydown",event=>{{if(event.target.matches("input,textarea,select"))return;if(event.key==="]")jump(1);if(event.key==="[")jump(-1);}}); render();
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def build_translation_artifacts(
    dataset_path: Path,
    output_dir: Path,
    translations_path: Path | None,
    sample_size: int,
    seed: int,
) -> None:
    dataset_hash = file_sha256(dataset_path)
    dataset = load_dataset(dataset_path)
    fixed_ids = None
    if translations_path is not None:
        payload = json.loads(translations_path.read_text(encoding="utf-8"))
        fixed_ids = [str(record["id"]) for record in payload["translations"]]
    sample, quotas = select_translation_subset(dataset, sample_size, seed, fixed_ids)
    report = build_sampling_report(dataset, sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_records = sample[SAMPLE_COLUMNS].where(
        pd.notna(sample[SAMPLE_COLUMNS]), None
    )
    source_payload = {
        "dataset_sha256": dataset_hash,
        "sample_size": sample_size,
        "seed": seed,
        "strata": ["group", "points", "visual_type"],
        "excluded_items": EXCLUDED_ITEMS,
        "selection_method": (
            "binary optimization matching proportional quotas for each year and each "
            "grade-point-visual-content cell; SHA-256 rank as deterministic objective"
        ),
        "items": source_records.to_dict(orient="records"),
    }
    (output_dir / "translation_source.json").write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    quotas.to_csv(output_dir / "sampling_quotas.csv", index=False)
    report.to_csv(output_dir / "sampling_report.csv", index=False)

    manifest = {
        "dataset_sha256": dataset_hash,
        "sample_size": sample_size,
        "seed": seed,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(sample["id"].astype(str))).encode("utf-8")
        ).hexdigest(),
        "sampling_strata": ["group", "points", "visual_type"],
        "excluded_items": EXCLUDED_ITEMS,
        "selection_method": (
            "binary optimization matching proportional quotas for each year and each "
            "grade-point-visual-content cell; SHA-256 rank as deterministic objective"
        ),
        "translations_included": translations_path is not None,
        "sample_fixed_by_translation_file": translations_path is not None,
        "post_review": POST_REVIEW,
    }

    if translations_path is not None:
        translated = load_translations(translations_path, sample)
        german, english = build_evaluation_frames(translated)
        german.to_parquet(output_dir / "german_control.parquet", index=False)
        english.to_parquet(output_dir / "english_translation.parquet", index=False)
        translated[
            SAMPLE_COLUMNS + [f"english_{field}" for field in TRANSLATED_FIELDS]
        ].to_csv(output_dir / "translation_subset.csv", index=False)
        render_review_html(translated, output_dir / "translation_review.html")
        manifest["translation_file_sha256"] = hashlib.sha256(
            translations_path.read_bytes()
        ).hexdigest()
        manifest["evaluation_design"] = (
            "paired German and English text; full German question crop omitted in both "
            "arms; extracted diagrams and image-based answer options retained"
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the representative German-English language-ablation subset"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/kangaroo.parquet"))
    parser.add_argument("--translations", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced/translation"),
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    build_translation_artifacts(
        args.dataset,
        args.output_dir,
        args.translations,
        args.sample_size,
        args.seed,
    )
    print(f"Built translation subset artifacts in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
