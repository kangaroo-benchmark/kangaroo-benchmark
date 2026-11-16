"""Option-order arm: the clean dataset with answer options in a random order and without the question crop.

The crop shows the printed options with their original letters, so it is omitted for every item
(statement text, associated figures and option content are kept). Options are permuted only where
their content does not depend on the letter: option images carry a printed letter, and some
statements or options refer to option letters; those items keep their order.
Usage: shuffle_options.py DATASET OUT_PARQUET OUT_MAPPING_CSV
"""

import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20261113
LETTERS = list("ABCDE")
LETTER_REFERENCE = re.compile(
    r"\(\s*[A-E]\s*\)|\b[A-E]\)"
    r"|\b(?:Antwort|Antworten|Option|Aussage|Aussagen|Angabe|Angaben|Bild|Figur|Möglichkeit)\s+\(?[A-E]\b"
    r"|\b[A-E]\s*(?:,|und|oder|bis)\s+[A-E]\b"
    r"|\b(?:obige[nr]?|vorherige[nr]?|vorstehende[nr]?|vorangehende[nr]?|vorangegangene[nr]?)\b"
)
LETTER_ONLY = re.compile(r"[\(\[]?[A-Ea-e][\)\]]?\.?")

source, out_parquet, out_mapping = sys.argv[1:4]
table = pq.read_table(source)
data = table.to_pandas()
texts = data[[f"sol_{x}" for x in LETTERS]].fillna("")
option_images = (
    data[[f"sol_{x}_image_bin" for x in LETTERS]]
    .map(lambda v: v is not None and len(v) > 0)
    .any(axis=1)
)
letter_only = texts.apply(lambda c: c.str.strip().str.fullmatch(LETTER_ONLY)).any(
    axis=1
)
refers = texts.apply(lambda c: c.map(lambda s: bool(LETTER_REFERENCE.search(s)))).any(
    axis=1
) | data["problem_statement"].fillna("").map(lambda s: bool(LETTER_REFERENCE.search(s)))
reason = np.select(
    [option_images, letter_only, refers],
    ["option images", "letter options", "letter reference"],
    "",
)

rng = np.random.default_rng(SEED)
shuffled = data.copy()
rows = []
for i, row in data.iterrows():
    order = (
        list(rng.permutation(5)) if reason[i] == "" else list(range(5))
    )  # order[new position] = old position
    for new, old in enumerate(order):
        shuffled.at[i, f"sol_{LETTERS[new]}"] = row[f"sol_{LETTERS[old]}"]
        shuffled.at[i, f"sol_{LETTERS[new]}_image_bin"] = row[
            f"sol_{LETTERS[old]}_image_bin"
        ]
    new_answer = LETTERS[order.index(LETTERS.index(row["answer"]))]
    shuffled.at[i, "answer"] = new_answer
    rows.append(
        {
            "id": row["id"],
            "kept_order_reason": reason[i],
            "old_answer": row["answer"],
            "new_answer": new_answer,
            **{
                f"new_{LETTERS[new]}_source": LETTERS[old]
                for new, old in enumerate(order)
            },
        }
    )
shuffled["question_image"] = None
pq.write_table(
    pa.Table.from_pandas(shuffled, schema=table.schema, preserve_index=False),
    out_parquet,
)
mapping = pd.DataFrame(rows)
mapping.to_csv(out_mapping, index=False)
print(mapping["kept_order_reason"].replace("", "shuffled").value_counts().to_dict())
moved = mapping[mapping["kept_order_reason"] == ""]
print(
    "shuffled items with the key in a new position:",
    int((moved.old_answer != moved.new_answer).sum()),
    "of",
    len(moved),
)
print(
    "key position after shuffle:",
    moved["new_answer"].value_counts().sort_index().to_dict(),
)
print(
    "key position before:      ",
    moved["old_answer"].value_counts().sort_index().to_dict(),
)
