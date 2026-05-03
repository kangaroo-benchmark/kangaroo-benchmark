(function(){
  function $(selector, root){
    return (root || document).querySelector(selector);
  }

  function $$(selector, root){
    return Array.from((root || document).querySelectorAll(selector));
  }

  function fetchJSON(url, params){
    if (params && Object.keys(params).length){
      var search = new URLSearchParams();
      Object.keys(params).forEach(function(key){
        var value = params[key];
        if (value === undefined || value === null || value === '') return;
        if (Array.isArray(value)){
          value.forEach(function(v){ search.append(key, v); });
        } else {
          search.append(key, value);
        }
      });
      if (url.indexOf('?') === -1){
        url += '?' + search.toString();
      } else {
        url += '&' + search.toString();
      }
    }
    return fetch(url, { headers: { 'Accept': 'application/json' } })
      .then(function(response){
        if (!response.ok){
          throw new Error('Request failed: ' + response.status);
        }
        return response.json();
      });
  }

  function saveTheme(theme){
    try {
      window.localStorage.setItem('dashboard:theme', theme);
    } catch (err){ /* ignore */ }
  }

  function loadTheme(){
    try {
      return window.localStorage.getItem('dashboard:theme');
    } catch (err){
      return null;
    }
  }

  function applyTheme(theme){
    var root = document.documentElement;
    if (theme === 'dark'){
      root.setAttribute('data-theme', 'dark');
    } else {
      root.setAttribute('data-theme', 'light');
    }
    saveTheme(theme);
  }

  var overviewListenersBound = false;

  function initThemeToggle(){
    var saved = loadTheme();
    if (saved){
      applyTheme(saved);
    }
    var toggle = $('#theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function(){
      var current = document.documentElement.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      document.dispatchEvent(new CustomEvent('dashboard:themechange', { detail: { theme: next } }));
    });
  }

  function initOverview(){
    var root = document.getElementById('overview-root');
    if (!root) return;

    if (!overviewListenersBound){
      document.body.addEventListener('htmx:beforeRequest', function(event){
        var current = document.getElementById('overview-root');
        if (!current) return;
        var source = event.detail && event.detail.elt;
        if (source && current.contains(source)){
          current.classList.add('is-loading');
        }
      });
      document.body.addEventListener('htmx:afterRequest', function(event){
        var current = document.getElementById('overview-root');
        if (!current) return;
        var source = event.detail && event.detail.elt;
        if (source && current.contains(source)){
          current.classList.remove('is-loading');
        }
      });
      document.body.addEventListener('htmx:afterSwap', function(event){
        if (event.target && event.target.id === 'overview-root'){
          event.target.classList.remove('is-loading');
          initOverview();
        }
      });
      overviewListenersBound = true;
    }

    var form = document.getElementById('overview-filters');
    if (!form) return;
    var toggleBtn = form.querySelector('[data-action="toggle-filters"]');
    var storageKey = 'dashboard:overview:filtersCollapsed';

    function setToggleState(collapsed){
      form.classList.toggle('is-collapsed', collapsed);
      if (toggleBtn){
        toggleBtn.textContent = collapsed ? 'Show advanced filters' : 'Hide advanced filters';
        toggleBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      }
    }

    var initial = true;
    try {
      var stored = window.localStorage.getItem(storageKey);
      if (stored !== null){
        initial = stored === '1';
      }
    } catch (err){
      /* ignore */
    }
    setToggleState(initial);

    if (toggleBtn && !toggleBtn.dataset.bound){
      toggleBtn.dataset.bound = 'true';
      toggleBtn.addEventListener('click', function(){
        var next = !form.classList.contains('is-collapsed');
        setToggleState(next);
        try {
          window.localStorage.setItem(storageKey, next ? '1' : '0');
        } catch (err){ /* ignore */ }
      });
    }

    if (!form.dataset.autosubmitBound){
      form.dataset.autosubmitBound = 'true';
      var submitDelay = 250;
      var submitTimer = null;
      var activeController = null;

      function submitWithFallback(){
        var hxGet = form.getAttribute('hx-get');
        var hxTarget = form.getAttribute('hx-target');
        if (!hxGet || !hxTarget || typeof window.fetch !== 'function'){
          return false;
        }

        if (activeController){
          activeController.abort();
        }
        var controller = new AbortController();
        activeController = controller;

        var rootEl = document.getElementById('overview-root');
        if (rootEl){
          rootEl.classList.add('is-loading');
        }

        var data = new FormData(form);
        var params = new URLSearchParams();
        data.forEach(function(value, key){
          params.append(key, value);
        });

        var url = hxGet;
        var query = params.toString();
        if (query){
          url += (url.indexOf('?') === -1 ? '?' : '&') + query;
        }
        var requestUrl = new URL(url, window.location.origin);

        fetch(requestUrl.toString(), {
          credentials: 'same-origin',
          headers: { 'HX-Request': 'true' },
          signal: controller.signal
        })
          .then(function(response){
            if (!response.ok){
              throw new Error('Request failed: ' + response.status);
            }
            return response.text().then(function(body){
              return { body: body, response: response };
            });
          })
          .then(function(payload){
            if (controller.signal.aborted){
              return;
            }
            var parser = new DOMParser();
            var doc = parser.parseFromString(payload.body, 'text/html');
            var replacement = doc.querySelector(hxTarget);
            var targetEl = document.querySelector(hxTarget);
            if (replacement && targetEl){
              targetEl.replaceWith(replacement);
              if (form.getAttribute('hx-push-url') === 'true'){
                var finalUrl = payload.response && payload.response.url ? new URL(payload.response.url) : requestUrl;
                window.history.pushState({}, '', finalUrl.pathname + finalUrl.search + finalUrl.hash);
              }
              var refreshed = document.querySelector('#overview-root');
              if (refreshed){
                refreshed.classList.remove('is-loading');
              }
              initOverview();
            }
          })
          .catch(function(err){
            if (err.name === 'AbortError'){
              return;
            }
            console.error('Overview refresh failed', err);
            if (typeof form.requestSubmit === 'function'){
              form.requestSubmit();
            } else {
              form.submit();
            }
          })
          .finally(function(){
            if (activeController === controller){
              activeController = null;
            }
            if (rootEl){
              rootEl.classList.remove('is-loading');
            }
          });

        return true;
      }

      function queueSubmit(){
        if (submitTimer){
          window.clearTimeout(submitTimer);
        }
        submitTimer = window.setTimeout(function(){
          if (window.htmx && typeof window.htmx.trigger === 'function' && window.htmx.version !== '0.0.1-local'){
            window.htmx.trigger(form, 'submit');
          } else if (submitWithFallback()){
            // handled via fetch fallback
          } else if (typeof form.requestSubmit === 'function'){
            form.requestSubmit();
          } else {
            form.submit();
          }
        }, submitDelay);
      }

      form.addEventListener('change', function(event){
        if (!event.target || !event.target.name) return;
        queueSubmit();
      });

      form.addEventListener('input', function(event){
        var target = event.target;
        if (!target || !target.name) return;
        if (target.matches('input[type="search"], input[type="text"], input[type="date"], input[type="number"], textarea')){
          queueSubmit();
        }
      });
    }
  }

  function initRunDetail(){
    var container = document.querySelector('.run-detail');
    if (!container) return;
    var bootstrap = $('#run-bootstrap');
    if (!bootstrap) return;
    var payload = {};
    try {
      payload = JSON.parse(bootstrap.textContent || '{}');
    } catch (err){
      console.error('Failed to parse run bootstrap', err);
      return;
    }
    var runId = payload.runId;
    if (!runId) return;

    var charts = new window.DashboardCharts();
    var lastAggregates = null;
    var state = {
      runId: runId,
      filters: {
        group: [],
        year: [],
        language: [],
        predicted: [],
        reasoning_mode: [],
        warning_types: [],
        warnings_present: '',
        multimodal: '',
        correctness: '',
        points_min: '',
        points_max: '',
        latency_min: '',
        latency_max: '',
        tokens_min: '',
        tokens_max: '',
        reasoning_tokens_min: '',
        reasoning_tokens_max: '',
        cost_min: '',
        cost_max: '',
        page: 1,
        page_size: 25,
        sort_by: 'id',
        sort_dir: 'asc'
      },
      humanComparison: null
    };

    var tableBody = $('#results-table tbody');
    var pagination = $('#results-pagination');
    var filterForm = $('#filters-form');
    var chipContainers = $$('.chip-select, select[data-facet]', filterForm);
    var sortBy = $('#sort-by');
    var sortDir = $('#sort-dir');
    var pageSize = $('#page-size');
    var resetBtn = $('#reset-filters');
    var exportJson = $('#export-json');
    var exportCsv = $('#export-csv');
    var warningList = $('#warning-toplist');
    var failuresList = $('#failures-timeline');
    var rowDrawer = $('#row-drawer');
    var rowMeta = $('#row-meta');
    var rowRationale = $('#row-rationale');
    var rowRaw = $('#row-raw');
    var rowDataset = $('#row-dataset');
    var rawToggle = $('#toggle-raw-response');
    var rawResponse = $('#raw-response');
    var closeRowBtn = $('#close-row');
    var activeFilters = $('#active-filters');
    var presetName = $('#preset-name');
    var savePresetBtn = $('#save-preset');
    var loadPresetBtn = $('#load-preset');
    var presetSelect = $('#preset-select');
    var subsetSummaryBody = $('#correctness-summary tbody');
    var subsetSummaryNote = $('#subset-summary-note');
    var subsetToggle = $('#subset-correctness-toggle');
    var gradeChart = $('#chart-grade-subset');
    var pointsChart = $('#chart-points-subset');
    var pointsEarnedChart = $('#chart-points-earned');
    var subsetLanguageList = $('#subset-language-list');
    var subsetReasoningList = $('#subset-reasoning-list');
    var subsetStatsList = $('#subset-stats-list');
    var subsetUIEnabled = Boolean(subsetSummaryBody || subsetToggle || gradeChart || pointsChart || pointsEarnedChart || subsetStatsList);
    var subsetMetrics = [];
    var selectedSubsetKey = null;
    var SUBSET_TABLE_ORDER = ['all','incorrect','correct','unknown'];
    var SUBSET_TOGGLE_ORDER = ['incorrect','correct','unknown','all'];
    var humanCards = $('#human-baseline-cards');
    var humanPercentileValue = $('#run-human-percentile');
    var humanPercentileNote = $('#run-human-percentile-note');
    var humanZScoreValue = $('#run-human-zscore');
    var humanZScoreNote = $('#run-human-zscore-note');
    var humanScoreValue = $('#run-human-score');
    var humanScoreNote = $('#run-human-score-note');
    var humanHelp = $('#human-baseline-help');

    function updatePresetSelect(){
      var presets = loadPresets();
      presetSelect.innerHTML = '';
      Object.keys(presets).sort().forEach(function(name){
        var option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        presetSelect.appendChild(option);
      });
    }

    function loadPresets(){
      try {
        var raw = window.localStorage.getItem('dashboard:presets:' + runId);
        return raw ? JSON.parse(raw) : {};
      } catch (err){
        return {};
      }
    }

    function savePresets(presets){
      try {
        window.localStorage.setItem('dashboard:presets:' + runId, JSON.stringify(presets));
      } catch (err){
        console.warn('Unable to save preset', err);
      }
    }

    function serializeFilters(){
      var params = Object.assign({}, state.filters);
      var arrays = ['group','year','language','predicted','reasoning_mode','warning_types'];
      arrays.forEach(function(key){
        params[key] = state.filters[key].slice();
      });
      return params;
    }

    function restoreFilters(data){
      Object.keys(state.filters).forEach(function(key){
        var value = data[key];
        if (Array.isArray(state.filters[key])){
          state.filters[key] = Array.isArray(value) ? value.slice() : [];
        } else {
          state.filters[key] = value ?? '';
        }
      });
      state.filters.page = 1;
      syncControls();
      refresh();
    }

    function facetToField(facet){
      switch (facet){
        case 'groups': return 'group';
        case 'years': return 'year';
        case 'languages': return 'language';
        case 'predicted_letters': return 'predicted';
        case 'reasoning_modes': return 'reasoning_mode';
        case 'warning_types': return 'warning_types';
        default: return facet;
      }
    }

    function syncControls(){
      // Chip selections
      chipContainers.forEach(function(container){
        var facet = container.dataset.facet;
        var field = facetToField(facet);
        if (container.tagName === 'SELECT'){
          Array.from(container.options).forEach(function(option){
            option.selected = state.filters[field].indexOf(option.value) !== -1;
          });
          return;
        }
        $$('.chip-select button', container).forEach(function(button){
          var value = button.dataset.value;
          var list = state.filters[field];
          if (Array.isArray(list) && list.indexOf(value) !== -1){
            button.classList.add('active');
          } else {
            button.classList.remove('active');
          }
        });
      });
      filterForm.multimodal.value = state.filters.multimodal;
      filterForm.correctness.value = state.filters.correctness;
      filterForm.warnings_present.value = state.filters.warnings_present;
      ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','reasoning_tokens_min','reasoning_tokens_max','cost_min','cost_max'].forEach(function(field){
        filterForm.elements[field].value = state.filters[field];
      });
      sortBy.value = state.filters.sort_by;
      sortDir.value = state.filters.sort_dir;
      pageSize.value = state.filters.page_size;
      renderActiveFilters();
    }

    function renderActiveFilters(){
      if (!activeFilters) return;
      var items = [];
      ['group','year','language','predicted','reasoning_mode','warning_types'].forEach(function(key){
        if (state.filters[key] && state.filters[key].length){
          items.push(key + ': ' + state.filters[key].join(', '));
        }
      });
      ['multimodal','correctness','warnings_present'].forEach(function(key){
        if (state.filters[key]){
          items.push(key + ': ' + state.filters[key]);
        }
      });
      ['points','latency','tokens','reasoning_tokens','cost'].forEach(function(prefix){
        var minKey = prefix + '_min';
        var maxKey = prefix + '_max';
        if (state.filters[minKey] || state.filters[maxKey]){
          items.push(prefix + ': ' + (state.filters[minKey] || '–') + '…' + (state.filters[maxKey] || '–'));
        }
      });
      activeFilters.textContent = items.join(' · ');
    }

    function formatPercent(value){
      if (value === null || value === undefined) return '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return '–';
      return (numeric * 100).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';
    }

    function formatNumber(value, decimals, fallback){
      if (value === null || value === undefined || value === '') return fallback || '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return fallback || '–';
      if (decimals === undefined){
        return numeric.toLocaleString();
      }
      return numeric.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    function formatCurrency(value){
      if (value === null || value === undefined || value === '') return '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return '–';
      return '$' + numeric.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
    }

    function describeSummary(summary, decimals, options){
      if (!summary || !summary.count){
        return 'No data';
      }
      var formatter = (options && options.formatter) || function(value){ return formatNumber(value, decimals); };
      var parts = [];
      if (summary.mean !== null && summary.mean !== undefined){
        var meanText = formatter(summary.mean);
        if (meanText && meanText !== '–'){
          parts.push('mean ' + meanText);
        }
      }
      if (summary.median !== null && summary.median !== undefined){
        var medianText = formatter(summary.median);
        if (medianText && medianText !== '–'){
          parts.push('median ' + medianText);
        }
      }
      if (summary.p25 !== null && summary.p25 !== undefined && summary.p75 !== null && summary.p75 !== undefined){
        var low = formatter(summary.p25);
        var high = formatter(summary.p75);
        if (low && high && low !== '–' && high !== '–'){
          parts.push('p25–p75 ' + low + '…' + high);
        }
      }
      if (!parts.length){
        return 'No data';
      }
      return parts.join(' · ');
    }

    function renderDistributionList(element, entries){
      if (!element) return;
      element.innerHTML = '';
      var list = entries || [];
      if (!list.length){
        var empty = document.createElement('li');
        empty.textContent = 'No data available.';
        element.appendChild(empty);
        return;
      }
      list.forEach(function(entry){
        var li = document.createElement('li');
        var name = document.createElement('span');
        name.textContent = entry && entry.label ? entry.label : 'Unknown';
        var value = document.createElement('span');
        var percentText = entry && entry.percentage !== undefined && entry.percentage !== null ? formatPercent(entry.percentage) : '–';
        var countText = entry && entry.count !== undefined && entry.count !== null ? entry.count : 0;
        value.textContent = countText + ' · ' + percentText;
        li.appendChild(name);
        li.appendChild(value);
        element.appendChild(li);
      });
    }

    function createStatItem(label, value){
      var li = document.createElement('li');
      var name = document.createElement('span');
      name.textContent = label;
      var detail = document.createElement('span');
      detail.textContent = value;
      li.appendChild(name);
      li.appendChild(detail);
      return li;
    }

    function renderSubsetStats(metric){
      if (!subsetStatsList) return;
      subsetStatsList.innerHTML = '';
      if (!metric || !metric.count){
        subsetStatsList.appendChild(createStatItem('Subset', 'No data for the current selection.'));
        return;
      }
      subsetStatsList.appendChild(createStatItem('Multimodal share', formatPercent(metric.multimodal_share)));
      subsetStatsList.appendChild(createStatItem('Points', describeSummary(metric.points_summary, 2)));
      if (metric.points_earned_summary && metric.points_earned_summary.count){
        subsetStatsList.appendChild(createStatItem('Points earned', describeSummary(metric.points_earned_summary, 2)));
      } else {
        subsetStatsList.appendChild(createStatItem('Points earned', 'No scoring data recorded.'));
      }
      subsetStatsList.appendChild(createStatItem('Latency', describeSummary(metric.latency_summary, 1, {
        formatter: function(value){ return formatNumber(value, 1) + ' ms'; }
      })));
      subsetStatsList.appendChild(createStatItem('Total tokens', describeSummary(metric.tokens_summary, 0)));
      subsetStatsList.appendChild(createStatItem('Cost', describeSummary(metric.cost_summary, 4, {
        formatter: function(value){ return formatCurrency(value); }
      })));
    }

    function findDefaultSubsetKey(metrics){
      if (!metrics || !metrics.length) return null;
      var order = ['incorrect', 'correct', 'all', 'unknown'];
      for (var i = 0; i < order.length; i++){
        var key = order[i];
        var match = metrics.find(function(item){ return item.key === key && item.count > 0; });
        if (match){
          return key;
        }
      }
      return metrics[0].key;
    }

    function updateSubsetSummary(){
      if (!subsetSummaryBody) return;
      subsetSummaryBody.innerHTML = '';
      if (!subsetMetrics.length){
        var emptyRow = document.createElement('tr');
        var emptyCell = document.createElement('td');
        emptyCell.colSpan = 10;
        emptyCell.textContent = 'No subset statistics available for the current filters.';
        emptyRow.appendChild(emptyCell);
        subsetSummaryBody.appendChild(emptyRow);
        return;
      }
      var totalMetric = subsetMetrics.find(function(item){ return item.key === 'all'; });
      if (!totalMetric || !totalMetric.count){
        var noneRow = document.createElement('tr');
        var noneCell = document.createElement('td');
        noneCell.colSpan = 10;
        noneCell.textContent = 'No rows match the current filters.';
        noneRow.appendChild(noneCell);
        subsetSummaryBody.appendChild(noneRow);
        return;
      }
      SUBSET_TABLE_ORDER.forEach(function(key){
        var metric = subsetMetrics.find(function(item){ return item.key === key; });
        if (!metric) return;
        var tr = document.createElement('tr');
        tr.dataset.key = metric.key;
        if (metric.key === selectedSubsetKey){
          tr.classList.add('is-active');
        }
        tr.innerHTML = [
          '<td>' + metric.label + '</td>',
          '<td>' + metric.count + '</td>',
          '<td>' + formatPercent(metric.share) + '</td>',
          '<td>' + formatPercent(metric.accuracy) + '</td>',
          '<td>' + formatPercent(metric.multimodal_share) + '</td>',
          '<td>' + formatNumber(metric.points_summary && metric.points_summary.mean, 2) + '</td>',
          '<td>' + formatNumber(metric.points_earned_summary && metric.points_earned_summary.mean, 2) + '</td>',
          '<td>' + formatNumber(metric.latency_summary && metric.latency_summary.median, 1) + '</td>',
          '<td>' + formatNumber(metric.tokens_summary && metric.tokens_summary.mean, 0) + '</td>',
          '<td>' + formatCurrency(metric.cost_summary && metric.cost_summary.mean) + '</td>'
        ].join('');
        tr.addEventListener('click', function(){ selectSubset(metric.key); });
        subsetSummaryBody.appendChild(tr);
      });
    }

    function updateSubsetToggle(){
      if (!subsetToggle) return;
      subsetToggle.innerHTML = '';
      if (!subsetMetrics.length) return;
      SUBSET_TOGGLE_ORDER.forEach(function(key){
        var metric = subsetMetrics.find(function(item){ return item.key === key; });
        if (!metric) return;
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = metric.label + ' (' + metric.count + ')';
        if (metric.key === selectedSubsetKey){
          btn.classList.add('is-active');
        }
        if (!metric.count){
          btn.disabled = true;
        }
        btn.addEventListener('click', function(){ selectSubset(metric.key); });
        subsetToggle.appendChild(btn);
      });
    }

    function updateSubsetCharts(){
      if (!subsetUIEnabled) return;
      var metric = null;
      if (selectedSubsetKey){
        metric = subsetMetrics.find(function(item){ return item.key === selectedSubsetKey; }) || null;
      }
      if (!metric){
        metric = subsetMetrics.find(function(item){ return item.key === 'all'; }) || null;
      }
      if (!metric){
        if (subsetSummaryNote){
          subsetSummaryNote.textContent = 'No subset statistics available.';
        }
        if (gradeChart){
          charts.updateDistribution(gradeChart, 'subset-grade', []);
        }
        if (pointsChart){
          charts.updateHistogram(pointsChart, {});
        }
        if (pointsEarnedChart){
          charts.updateHistogram(pointsEarnedChart, {});
        }
        renderDistributionList(subsetLanguageList, []);
        renderDistributionList(subsetReasoningList, []);
        renderSubsetStats(null);
        return;
      }
      if (subsetSummaryNote){
        if (!metric.count){
          subsetSummaryNote.textContent = metric.label + ': no rows for the current filters.';
        } else {
          subsetSummaryNote.textContent = metric.label + ' · ' + metric.count + ' rows (' + formatPercent(metric.share) + ' of filtered)';
        }
      }
      if (gradeChart){
        charts.updateDistribution(gradeChart, 'subset-grade', metric.grade_distribution || []);
      }
      if (pointsChart){
        charts.updateHistogram(pointsChart, metric.points_hist || {});
      }
      if (pointsEarnedChart){
        charts.updateHistogram(pointsEarnedChart, metric.points_earned_hist || {});
      }
      renderDistributionList(subsetLanguageList, metric.language_distribution || []);
      renderDistributionList(subsetReasoningList, metric.reasoning_mode_distribution || []);
      renderSubsetStats(metric);
    }

    function refreshSubsetUI(){
      if (!subsetUIEnabled) return;
      updateSubsetSummary();
      updateSubsetToggle();
      updateSubsetCharts();
    }

    function selectSubset(key){
      if (!subsetUIEnabled) return;
      if (!key || key === selectedSubsetKey){
        return;
      }
      selectedSubsetKey = key;
      refreshSubsetUI();
    }

    function setSubsetMetrics(metrics){
      if (!subsetUIEnabled) return;
      subsetMetrics = Array.isArray(metrics) ? metrics.slice() : [];
      if (!subsetMetrics.length){
        selectedSubsetKey = null;
        refreshSubsetUI();
        return;
      }
      if (!selectedSubsetKey || !subsetMetrics.some(function(item){ return item.key === selectedSubsetKey && item.count > 0; })){
        selectedSubsetKey = findDefaultSubsetKey(subsetMetrics);
      }
      refreshSubsetUI();
    }

    function attachChipHandlers(){
      chipContainers.forEach(function(container){
        if (container.tagName === 'SELECT'){
          container.addEventListener('change', function(){
            var facet = container.dataset.facet;
            var field = facetToField(facet);
            state.filters[field] = Array.from(container.selectedOptions).map(function(option){ return option.value; });
            state.filters.page = 1;
            refresh();
          });
          return;
        }
        container.addEventListener('click', function(event){
          var target = event.target;
          if (target.tagName !== 'BUTTON') return;
          var facet = container.dataset.facet;
          var field = facetToField(facet);
          var list = state.filters[field];
          var value = target.dataset.value;
          var index = list.indexOf(value);
          if (index === -1){
            list.push(value);
            target.classList.add('active');
          } else {
            list.splice(index, 1);
            target.classList.remove('active');
          }
          state.filters.page = 1;
          refresh();
        });
      });
    }

    function buildQuery(){
      var params = {};
      Object.keys(state.filters).forEach(function(key){
        var value = state.filters[key];
        if (Array.isArray(value)){
          if (value.length){
            params[key] = value;
          }
        } else if (value !== '' && value !== null && value !== undefined){
          params[key] = value;
        }
      });
      params.page = state.filters.page;
      params.page_size = state.filters.page_size;
      params.sort_by = state.filters.sort_by;
      params.sort_dir = state.filters.sort_dir;
      return params;
    }

    function refresh(){
      renderActiveFilters();
      loadResults();
      loadAggregates();
    }

    function loadFacets(){
      fetchJSON('/api/runs/' + runId + '/facets')
        .then(function(facets){
          chipContainers.forEach(function(container){
            var facet = container.dataset.facet;
            var field = facetToField(facet);
            var values = facets[facet] || [];
            if (container.tagName === 'SELECT'){
              container.innerHTML = '';
              values.forEach(function(value){
                var option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                container.appendChild(option);
              });
            } else {
              container.innerHTML = '';
              values.forEach(function(value){
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = value;
                btn.dataset.value = value;
                container.appendChild(btn);
              });
            }
          });
          attachChipHandlers();
          syncControls();
        })
        .catch(function(err){ console.error('Failed to load facets', err); });
    }

    function loadResults(){
      fetchJSON('/api/runs/' + runId + '/results', buildQuery())
        .then(function(page){
          renderTable(page.items || []);
          renderPagination(page.page, page.page_size, page.total);
        })
        .catch(function(err){
          console.error('Failed to load results', err);
        });
    }

    function renderTable(items){
      tableBody.innerHTML = '';
      if (!items.length){
        var empty = document.createElement('tr');
        var cell = document.createElement('td');
        cell.colSpan = 14;
        cell.textContent = 'No rows match the current filters.';
        empty.appendChild(cell);
        tableBody.appendChild(empty);
        return;
      }
      items.forEach(function(row){
        var tr = document.createElement('tr');
        tr.innerHTML = [
          row.id,
          row.group || '–',
          row.year || '–',
          row.problem_number || '–',
          row.points ?? '–',
          row.answer || '–',
          row.predicted || '–',
          row.is_correct === true ? '✔' : row.is_correct === false ? '✖' : '–',
          row.points_earned ?? '–',
          row.latency_ms ? row.latency_ms.toFixed(1) : '–',
          row.total_tokens ?? '–',
          row.reasoning_tokens ?? '–',
          row.cost_usd ? row.cost_usd.toFixed(4) : '–',
          row.warnings && row.warnings.length ? '⚠ ' + row.warnings.length : '–'
        ].map(function(value){ return '<td>' + value + '</td>'; }).join('');
        tr.addEventListener('click', function(){ openRow(row.id); });
        tableBody.appendChild(tr);
      });
    }

    function renderPagination(page, pageSizeValue, total){
      pagination.innerHTML = '';
      var pageCount = Math.max(1, Math.ceil(total / pageSizeValue));

      var createButton = function(label, newPage, options){
        var opts = options || {};
        var btn = document.createElement('button');
        btn.textContent = label;
        if (opts.isEllipsis){
          btn.disabled = true;
          btn.classList.add('ellipsis');
        } else if (opts.disabled){
          btn.disabled = true;
        } else {
          btn.addEventListener('click', function(){
            state.filters.page = newPage;
            refresh();
          });
        }
        if (opts.active){
          btn.classList.add('active');
        }
        pagination.appendChild(btn);
      };

      createButton('Prev', Math.max(1, page - 1), { disabled: page <= 1 });

      var windowRadius = 2;
      var start = Math.max(2, page - windowRadius);
      var end = Math.min(pageCount - 1, page + windowRadius);

      var addPage = function(pageNumber){
        createButton(String(pageNumber), pageNumber, { active: pageNumber === page });
      };

      addPage(1);

      if (start > 2){
        createButton('…', null, { isEllipsis: true });
      }

      for (var i = start; i <= end; i++){
        addPage(i);
      }

      if (end < pageCount - 1){
        createButton('…', null, { isEllipsis: true });
      }

      if (pageCount > 1){
        addPage(pageCount);
      }

      createButton('Next', Math.min(pageCount, page + 1), { disabled: page >= pageCount });
    }

    function loadAggregates(){
      fetchJSON('/api/runs/' + runId + '/aggregates', buildQuery())
        .then(function(data){
          lastAggregates = data;
          if (subsetUIEnabled){
            setSubsetMetrics(data.subset_metrics || []);
          }
          charts.updateBreakdown($('#chart-group'), 'chart-group', data.breakdown_by_group || {});
          charts.updateBreakdown($('#chart-year'), 'chart-year', data.breakdown_by_year || {});
          charts.updateConfusion($('#chart-confusion'), data.confusion_matrix || {});
          charts.updateHistogram($('#chart-latency'), data.latency_hist || {});
          charts.updateHistogram($('#chart-token'), data.tokens_hist || {});
          charts.updatePredicted($('#chart-predicted'), data.predicted_counts || {});
          renderWarningList(data.warning_toplist || []);
        })
        .catch(function(err){ console.error('Failed to load aggregates', err); });
    }

    function updateHumanCards(entry){
      if (!humanCards) return;
      if (!entry){
        humanCards.hidden = true;
        if (humanHelp){
          humanHelp.hidden = true;
        }
        return;
      }
      humanCards.hidden = false;
      if (humanHelp){
        humanHelp.hidden = false;
        humanHelp.textContent = 'Top grade (by human percentile). LLM totals include the official start capital and -¼ penalties for wrong or unparsed answers.';
      }
      var gradeLabel = entry.grade_label || entry.grade_id || '';
      var gradeDisplay = gradeLabel ? 'Grade ' + gradeLabel : (entry.grade_id ? 'Grade ' + entry.grade_id : '');
      if (humanPercentileValue){
        humanPercentileValue.textContent = formatPercent(entry.human_percentile);
      }
      if (humanPercentileNote){
        humanPercentileNote.textContent = gradeDisplay;
      }
      if (humanZScoreValue){
        if (entry.z_score !== null && entry.z_score !== undefined){
          humanZScoreValue.textContent = entry.z_score.toFixed(2);
        } else {
          humanZScoreValue.textContent = '–';
        }
      }
      if (humanZScoreNote){
        var zParts = [];
        if (gradeDisplay){
          zParts.push(gradeDisplay);
        }
        if (entry.human_mean !== null && entry.human_mean !== undefined){
          zParts.push('μ ≈ ' + formatNumber(entry.human_mean, 1));
        }
        if (entry.human_std){
          zParts.push('σ ≈ ' + formatNumber(entry.human_std, 2));
        }
        humanZScoreNote.textContent = zParts.join(' • ');
      }
      if (humanScoreValue){
        if (entry.llm_total !== null && entry.llm_max !== null){
          humanScoreValue.textContent = formatNumber(entry.llm_total, 1) + ' / ' + formatNumber(entry.llm_max, 1);
        } else {
          humanScoreValue.textContent = '–';
        }
      }
      if (humanScoreNote){
        var scoreParts = [];
        if (gradeDisplay){
          scoreParts.push(gradeDisplay);
        }
        if (entry.llm_start_points !== null && entry.llm_start_points !== undefined){
          scoreParts.push('Start ' + formatNumber(entry.llm_start_points, 0));
        }
        var runMax = 0;
        if (entry.llm_points_available !== null && entry.llm_points_available !== undefined){
          runMax = Number(entry.llm_points_available);
        }
        if (entry.llm_start_points !== null && entry.llm_start_points !== undefined){
          runMax += Number(entry.llm_start_points);
        }
        if (runMax){
          scoreParts.push('Run max ' + formatNumber(runMax, 1));
        }
        if (entry.max_points !== null && entry.max_points !== undefined){
          scoreParts.push('Official max ' + formatNumber(entry.max_points, 0));
        }
        humanScoreNote.textContent = scoreParts.join(' • ');
      }
    }

    function loadHumanComparison(){
      var url = '/api/humans/compare/run/' + runId + '?late_year_strategy=' + encodeURIComponent(state.aggregationStrategy);
      fetchJSON(url)
        .then(function(payload){
          state.humanComparison = payload;
          if (!payload || !payload.entries || !payload.entries.length){
            updateHumanCards(null);
            return;
          }
          var sorted = payload.entries.slice().sort(function(a, b){
            var pa = a.human_percentile || 0;
            var pb = b.human_percentile || 0;
            return pb - pa;
          });
          updateHumanCards(sorted[0]);
        })
        .catch(function(){ updateHumanCards(null); });
    }

    function rerenderCharts(){
      if (!lastAggregates) return;
      charts.updateBreakdown($('#chart-group'), 'chart-group', lastAggregates.breakdown_by_group || {});
      charts.updateBreakdown($('#chart-year'), 'chart-year', lastAggregates.breakdown_by_year || {});
      charts.updateConfusion($('#chart-confusion'), lastAggregates.confusion_matrix || {});
      charts.updateHistogram($('#chart-latency'), lastAggregates.latency_hist || {});
      charts.updateHistogram($('#chart-token'), lastAggregates.tokens_hist || {});
      charts.updatePredicted($('#chart-predicted'), lastAggregates.predicted_counts || {});
      if (subsetUIEnabled){
        updateSubsetCharts();
      }
    }

    function renderWarningList(entries){
      if (!warningList) return;
      warningList.innerHTML = '';
      if (!entries.length){
        var li = document.createElement('li');
        li.textContent = 'No warnings recorded.';
        warningList.appendChild(li);
        return;
      }
      entries.forEach(function(entry){
        var li = document.createElement('li');
        li.textContent = entry.warning_type + ' · ' + entry.count;
        li.addEventListener('click', function(){
          if (state.filters.warning_types.indexOf(entry.warning_type) === -1){
            state.filters.warning_types.push(entry.warning_type);
            state.filters.warnings_present = 'true';
            syncControls();
            refresh();
          }
        });
        warningList.appendChild(li);
      });
    }

    function loadFailures(){
      fetchJSON('/api/runs/' + runId + '/failures')
        .then(function(entries){
          failuresList.innerHTML = '';
          if (!entries.length){
            var li = document.createElement('li');
            li.textContent = 'No failures recorded.';
            failuresList.appendChild(li);
            return;
          }
          entries.forEach(function(entry){
            var li = document.createElement('li');
            var summary = (entry.timestamp || 'n/a') + ' · ' + (entry.status_code || 'code?') + ' · ' + (entry.message || '');
            var summarySpan = document.createElement('span');
            summarySpan.textContent = summary;
            li.appendChild(summarySpan);
            if (entry.id){
              li.appendChild(document.createTextNode(' '));
              var link = document.createElement('button');
              link.type = 'button';
              link.className = 'link-button';
              link.textContent = 'View ' + entry.id;
              link.addEventListener('click', function(){
                openRow(entry.id);
              });
              li.appendChild(link);
            }
            failuresList.appendChild(li);
          });
        })
        .catch(function(err){ console.error('Failed to load failures', err); });
    }

    function openRow(rowId){
      fetchJSON('/api/runs/' + runId + '/row/' + encodeURIComponent(rowId))
        .then(function(payload){
          renderRowDrawer(payload);
        })
        .catch(function(err){
          console.error('Failed to load row detail', err);
        });
    }

    function renderRowDrawer(payload){
      rowMeta.innerHTML = '';
      var row = payload.row;
      var dataset = payload.dataset;
      var info = document.createElement('div');
      info.innerHTML = [
        '<strong>ID:</strong> ' + row.id,
        '<strong>Group:</strong> ' + (row.group || '–'),
        '<strong>Year:</strong> ' + (row.year || '–'),
        '<strong>Answer:</strong> ' + (row.answer || '–'),
        '<strong>Predicted:</strong> ' + (row.predicted || '–'),
        '<strong>Latency:</strong> ' + (row.latency_ms ? row.latency_ms.toFixed(1) + ' ms' : '–'),
        '<strong>Total tokens:</strong> ' + (row.total_tokens ?? '–'),
        '<strong>Reasoning tokens:</strong> ' + (row.reasoning_tokens ?? '–'),
        '<strong>Cost:</strong> ' + (row.cost_usd ? row.cost_usd.toFixed(4) : '–'),
        row.warnings && row.warnings.length ? '<strong>Warnings:</strong> ' + row.warnings.join(', ') : ''
      ].filter(Boolean).map(function(line){ return '<p>' + line + '</p>'; }).join('');
      rowMeta.appendChild(info);
      if (rowRationale){
        if (row.rationale){
          rowRationale.classList.remove('hidden');
          rowRationale.innerHTML = '<h3>Rationale</h3>';
          var rationaleBlock = document.createElement('pre');
          rationaleBlock.className = 'row-rationale__text';
          rationaleBlock.textContent = row.rationale;
          rowRationale.appendChild(rationaleBlock);
        } else {
          rowRationale.classList.add('hidden');
          rowRationale.innerHTML = '';
        }
      }
      if (rowRaw && rawToggle && rawResponse){
        if (row.raw_text_response){
          rowRaw.classList.remove('hidden');
          rawToggle.classList.remove('hidden');
          rawResponse.textContent = row.raw_text_response;
          rawResponse.classList.add('hidden');
          rawToggle.textContent = 'Show raw response';
          rawToggle.onclick = function(){
            var isHidden = rawResponse.classList.toggle('hidden');
            rawToggle.textContent = isHidden ? 'Show raw response' : 'Hide raw response';
          };
        } else {
          rowRaw.classList.add('hidden');
          rawToggle.classList.add('hidden');
          rawResponse.textContent = '';
          rawResponse.classList.add('hidden');
          rawToggle.onclick = null;
        }
      }
      rowDataset.innerHTML = '';
      if (dataset){
        var problem = document.createElement('article');
        problem.innerHTML = '<h3>Problem statement</h3><p>' + (dataset.problem_statement || 'n/a') + '</p>';
        rowDataset.appendChild(problem);
        if (dataset.question_image){
          var img = document.createElement('img');
          img.src = dataset.question_image;
          img.alt = 'Question image';
          rowDataset.appendChild(img);
        }
        if (dataset.options && dataset.options.length){
          var list = document.createElement('ul');
          dataset.options.forEach(function(option){
            var item = document.createElement('li');
            var content = '<strong>' + option.label + '.</strong> ' + (option.text || '');
            if (option.image){
              content += '<br><img src="' + option.image + '" alt="Option ' + option.label + '" />';
            }
            item.innerHTML = content;
            list.appendChild(item);
          });
          rowDataset.appendChild(list);
        }
        if (dataset.associated_images && dataset.associated_images.length){
          dataset.associated_images.forEach(function(src){
            var img = document.createElement('img');
            img.src = src;
            img.alt = 'Associated image';
            rowDataset.appendChild(img);
          });
        }
      }
      rowDrawer.classList.remove('hidden');
      rowDrawer.classList.add('visible');
    }

    function closeDrawer(){
      rowDrawer.classList.remove('visible');
      rowDrawer.classList.add('hidden');
    }

    if (closeRowBtn){
      closeRowBtn.addEventListener('click', closeDrawer);
    }

    function attachSelectChange(select, key){
      if (!select) return;
      select.addEventListener('change', function(){
        state.filters[key] = select.value;
        state.filters.page = 1;
        refresh();
      });
    }

    function attachRangeInput(field){
      var input = filterForm.elements[field];
      if (!input) return;
      input.addEventListener('change', function(){
        state.filters[field] = input.value;
        state.filters.page = 1;
        refresh();
      });
    }

    filterForm.addEventListener('submit', function(event){
      event.preventDefault();
      state.filters.multimodal = filterForm.multimodal.value;
      state.filters.correctness = filterForm.correctness.value;
      state.filters.warnings_present = filterForm.warnings_present.value;
      ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','cost_min','cost_max'].forEach(function(field){
        state.filters[field] = filterForm.elements[field].value;
      });
      state.filters.page = 1;
      refresh();
    });

    attachSelectChange(filterForm.multimodal, 'multimodal');
    attachSelectChange(filterForm.correctness, 'correctness');
    attachSelectChange(filterForm.warnings_present, 'warnings_present');
    ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','cost_min','cost_max'].forEach(attachRangeInput);

    sortBy.addEventListener('change', function(){
      state.filters.sort_by = sortBy.value;
      refresh();
    });

    sortDir.addEventListener('change', function(){
      state.filters.sort_dir = sortDir.value;
      refresh();
    });

    pageSize.addEventListener('change', function(){
      state.filters.page_size = parseInt(pageSize.value, 10) || 25;
      state.filters.page = 1;
      refresh();
    });

    resetBtn.addEventListener('click', function(){
      Object.keys(state.filters).forEach(function(key){
        if (Array.isArray(state.filters[key])){
          state.filters[key] = [];
        } else if (['page','page_size'].indexOf(key) !== -1){
          // preserve pagination defaults
        } else if (key === 'sort_by'){
          state.filters[key] = 'id';
        } else if (key === 'sort_dir'){
          state.filters[key] = 'asc';
        } else if (key === 'page_size'){
          state.filters[key] = 25;
        } else {
          state.filters[key] = '';
        }
      });
      state.filters.page = 1;
      state.filters.page_size = 25;
      syncControls();
      refresh();
    });

    if (exportJson){
      exportJson.addEventListener('click', function(){
        triggerDownload('json');
      });
    }
    if (exportCsv){
      exportCsv.addEventListener('click', function(){
        triggerDownload('csv');
      });
    }

    function triggerDownload(format){
      var params = buildQuery();
      params.download = format;
      var query = new URLSearchParams();
      Object.keys(params).forEach(function(key){
        var value = params[key];
        if (Array.isArray(value)){
          value.forEach(function(v){ query.append(key, v); });
        } else {
          query.append(key, value);
        }
      });
      window.location.href = '/api/runs/' + runId + '/results?' + query.toString();
    }

    if (savePresetBtn){
      savePresetBtn.addEventListener('click', function(){
        var name = (presetName.value || '').trim();
        if (!name){
          alert('Enter a preset name');
          return;
        }
        var presets = loadPresets();
        presets[name] = serializeFilters();
        savePresets(presets);
        updatePresetSelect();
      });
    }

    if (loadPresetBtn){
      loadPresetBtn.addEventListener('click', function(){
        var name = presetSelect.value;
        if (!name) return;
        var presets = loadPresets();
        if (presets[name]){
          restoreFilters(presets[name]);
        }
      });
    }

    updatePresetSelect();
    loadFacets();
    loadFailures();
    syncControls();
    refresh();
    loadHumanComparison();

    document.addEventListener('dashboard:themechange', rerenderCharts);
  }

  function initCompare(){
    var compareSection = document.querySelector('.compare');
    if (!compareSection) return;
    var runAInput = $('#compare-run-a');
    var runBInput = $('#compare-run-b');
    var form = $('#compare-form');
    var viewSelect = $('#compare-view');
    var limitInput = $('#compare-limit');
    var refreshBtn = $('#compare-refresh');
    var compareTableBody = $('#compare-table tbody');
    var charts = new window.DashboardCharts();
    var lastCompare = null;
    var confusionToggle = $('#compare-confusion-toggle');
    var confusionSection = $('#compare-confusion-section');
    var confusionLabelA = $('#compare-confusion-label-a');
    var confusionLabelB = $('#compare-confusion-label-b');
    var confusionCanvasA = $('#compare-confusion-a');
    var confusionCanvasB = $('#compare-confusion-b');

    function redirect(){
      var params = new URLSearchParams();
      params.append('run_a', runAInput.value);
      params.append('run_b', runBInput.value);
      window.location.href = '/compare?' + params.toString();
    }

    if (form){
      form.addEventListener('submit', function(event){
        event.preventDefault();
        if (!runAInput.value || !runBInput.value) return;
        redirect();
      });
    }

    function fetchCompare(){
      if (!runAInput.value || !runBInput.value) return;
      var params = {
        run_a: runAInput.value,
        run_b: runBInput.value,
        view: viewSelect ? viewSelect.value : 'all'
      };
      var limitVal = limitInput && limitInput.value ? parseInt(limitInput.value, 10) : null;
      if (limitVal){
        params.limit = limitVal;
      }
      if (confusionToggle && confusionToggle.checked){
        params.include_confusion = 'true';
      } else if (confusionSection){
        confusionSection.classList.add('hidden');
      }
      fetchJSON('/api/compare', params)
        .then(function(payload){
          renderCompare(payload);
        })
        .catch(function(err){ console.error('Failed to fetch compare', err); });
    }

    function renderCompare(payload){
      lastCompare = payload;
      var metricGrid = $('#compare-metrics');
      if (metricGrid && payload.metrics){
        metricGrid.innerHTML = '';
        payload.metrics.forEach(function(metric){
          var deltaDisplay = '–';
          if (metric.delta !== null && metric.delta !== undefined){
            var deltaNumber = Number(metric.delta);
            deltaDisplay = isNaN(deltaNumber) ? '–' : deltaNumber.toFixed(4);
          }
          var card = document.createElement('div');
          card.className = 'summary-card';
          card.innerHTML = '<span class="summary-label">' + metric.metric.replace(/_/g, ' ') + '</span>' +
            '<span class="summary-value">' + deltaDisplay + '</span>' +
            '<span class="summary-note">A: ' + (metric.run_a ?? '–') + ' · B: ' + (metric.run_b ?? '–') + '</span>';
          metricGrid.appendChild(card);
        });
      }
      if (payload.breakdown_deltas){
        charts.updateCompareSeries($('#compare-group'), payload.breakdown_deltas.group || []);
        charts.updateCompareSeries($('#compare-year'), payload.breakdown_deltas.year || []);
      }
      if (payload.row_deltas && compareTableBody){
        compareTableBody.innerHTML = '';
        payload.row_deltas.forEach(function(row){
          var tr = document.createElement('tr');
          tr.innerHTML = [
            row.id,
            row.run_a_correct,
            row.run_b_correct,
            row.run_a_predicted || '–',
            row.run_b_predicted || '–',
            row.run_a_points ?? '–',
            row.run_b_points ?? '–',
            row.delta_points ?? '–',
            row.delta_latency_ms ?? '–',
            row.delta_total_tokens ?? '–'
          ].map(function(value){ return '<td>' + value + '</td>'; }).join('');
          compareTableBody.appendChild(tr);
        });
      }
      if (confusionSection){
        var hasConfusion = payload.confusion_matrices && payload.confusion_matrices.run_a && payload.confusion_matrices.run_b;
        if (confusionToggle && confusionToggle.checked && hasConfusion){
          confusionSection.classList.remove('hidden');
          if (confusionLabelA){
            confusionLabelA.textContent = payload.run_a.model_label || payload.run_a.model_id || payload.run_a.run_id;
          }
          if (confusionLabelB){
            confusionLabelB.textContent = payload.run_b.model_label || payload.run_b.model_id || payload.run_b.run_id;
          }
          if (confusionCanvasA){
            charts.updateConfusion(confusionCanvasA, payload.confusion_matrices.run_a);
          }
          if (confusionCanvasB){
            charts.updateConfusion(confusionCanvasB, payload.confusion_matrices.run_b);
          }
        } else {
          confusionSection.classList.add('hidden');
        }
      }
    }

    function rerenderCompareCharts(){
      if (!lastCompare) return;
      if (lastCompare.breakdown_deltas){
        charts.updateCompareSeries($('#compare-group'), lastCompare.breakdown_deltas.group || []);
        charts.updateCompareSeries($('#compare-year'), lastCompare.breakdown_deltas.year || []);
      }
      if (!confusionSection) return;
      var hasConfusion = lastCompare.confusion_matrices && lastCompare.confusion_matrices.run_a && lastCompare.confusion_matrices.run_b;
      if (confusionToggle && confusionToggle.checked && hasConfusion){
        confusionSection.classList.remove('hidden');
        if (confusionLabelA){
          confusionLabelA.textContent = lastCompare.run_a.model_label || lastCompare.run_a.model_id || lastCompare.run_a.run_id;
        }
        if (confusionLabelB){
          confusionLabelB.textContent = lastCompare.run_b.model_label || lastCompare.run_b.model_id || lastCompare.run_b.run_id;
        }
        charts.updateConfusion(confusionCanvasA, lastCompare.confusion_matrices.run_a);
        charts.updateConfusion(confusionCanvasB, lastCompare.confusion_matrices.run_b);
      }
    }

    if (refreshBtn){
      refreshBtn.addEventListener('click', fetchCompare);
    }
    if (viewSelect){
      viewSelect.addEventListener('change', fetchCompare);
    }
    if (limitInput){
      limitInput.addEventListener('change', fetchCompare);
    }
    if (confusionToggle){
      confusionToggle.addEventListener('change', function(){
        if (confusionToggle.checked){
          fetchCompare();
        } else if (confusionSection){
          confusionSection.classList.add('hidden');
        }
      });
    }

    // Auto fetch if selections present
    if (runAInput.value && runBInput.value){
      fetchCompare();
    }

    document.addEventListener('dashboard:themechange', rerenderCompareCharts);
  }

  function initAnalysis(){
    var analysisSection = document.querySelector('.analysis');
    if (!analysisSection) return;

    var form = $('#analysis-form');
    var resultsContainer = $('#analysis-results-container');
    var chartsContainer = $('.analysis-charts');
    var tagsContainer = $('.analysis-tags');
    var scatterPlotEl = $('#scatter-human-vs-llm');
    var barChartEl = $('#bar-normalized-performance');
    var tagTableBody = $('#tag-analysis-table tbody');
    var charts = new window.DashboardCharts();

    form.addEventListener('submit', function(event){
      event.preventDefault();
      var formData = new FormData(form);
      var runIds = formData.getAll('run_ids');
      if (!runIds.length) {
        alert('Please select at least one run.');
        return;
      }

      resultsContainer.style.display = 'block';
      $('#analysis-results').innerHTML = '<p>Loading analysis...</p>';
      chartsContainer.style.display = 'none';
      tagsContainer.style.display = 'none';

      // Construct form data for POST request
      var postData = new URLSearchParams();
      runIds.forEach(function(id) { postData.append('run_ids', id); });

      fetch('/api/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: postData
      })
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Request failed: ' + response.status);
        }
        return response.json();
      })
      .then(function(data){
          $('#analysis-results').innerHTML = ''; // Clear loading message
          chartsContainer.style.display = 'block';
          tagsContainer.style.display = 'block';
          
          var scatterData = data.scatter.map(function(item) {
            return [item.human_p_correct, item.avg_llm_score, item.question_id, item.llm_disagreement];
          }).filter(function(item) {
            return item[0] !== null && item[0] !== undefined;
          });

          charts.updateScatter(scatterPlotEl, scatterData);
          charts.updateBar(barChartEl, data.bars);
          renderTagTable(data.tags);
        })
        .catch(function(err){
          $('#analysis-results').innerHTML = '<p class="error">Failed to load analysis: ' + err.message + '</p>';
          console.error('Failed to load analysis', err);
        });
    });

    function renderTagTable(tags) {
      if (!tagTableBody) return;
      tagTableBody.innerHTML = '';
      if (!tags || !tags.length) {
        var row = tagTableBody.insertRow();
        var cell = row.insertCell();
        cell.colSpan = 4;
        cell.textContent = 'No tags found for this selection.';
        return;
      }

      tags.forEach(function(tag) {
        var row = tagTableBody.insertRow();
        row.insertCell().textContent = tag.tag;
        row.insertCell().textContent = tag.count;
        row.insertCell().textContent = (tag.avg_human_score * 100).toFixed(1) + '%';
        row.insertCell().textContent = (tag.avg_llm_score * 100).toFixed(1) + '%';
      });
    }
  }

  function initHumans(){
    var root = document.getElementById('human-baseline-root');
    if (!root) return;
    if (root.dataset.hasData !== 'true') return;

    var bootstrapEl = document.getElementById('human-baseline-bootstrap');
    var bootstrap = { runs: [], years: [] };
    if (bootstrapEl && bootstrapEl.textContent){
      try {
        bootstrap = JSON.parse(bootstrapEl.textContent);
      } catch (err){
        console.error('Failed to parse human baseline bootstrap', err);
      }
    }

    var charts = new window.DashboardCharts();
    var state = {
      runs: bootstrap.runs || [],
      years: bootstrap.years || [],
      yearMap: {},
      selectedView: 'run',
      selectedRun: null,
      selectedRuns: [],
      selectedYear: null,
      selectedGrade: null,
      selectedTableView: 'all-years',
      selectedGroupingMode: 'year',
      selectedCohortType: 'micro',
      selectedComparisonSource: 'average', // 'average' or 'best'
      aggregationStrategy: 'best',
      runComparisons: {},
      runSummaries: {},
      cohortCache: {},
      cohortSummaries: {},
      yearSummaryCache: {},
      cdfCache: {},
      percentileCache: {},
      baselineSummaries: {},
      currentRunSummary: null,
      currentRunSummaryKey: null,
      runTableSortOverride: null,
    };

    state.years.forEach(function(entry){
      if (entry && entry.year !== undefined){
        state.yearMap[entry.year] = entry;
      }
    });

    var viewSelect = $('#human-view-select');
    var runSelect = $('#human-run-select');
    var runSelectWrapper = $('#human-run-select-wrapper');
    var runChecklist = $('#human-run-checklist');
    var runMultiWrapper = $('#human-run-multi-wrapper');
    var runSelectAllBtn = $('#human-run-select-all');
    var runClearBtn = $('#human-run-clear');
    var groupingModeSelect = $('#human-grouping-mode');
    var yearSelect = $('#human-year-select');
    var yearSelectWrapper = $('#human-year-select-wrapper');
    var gradeSelect = $('#human-grade-select');
    var gradeSelectWrapper = $('#human-grade-select-wrapper');
    var tableViewSelect = $('#human-table-view');
    var cohortTypeSelect = $('#human-cohort-type');
    var cohortTypeWrapper = $('#human-cohort-type-wrapper');
    var comparisonSourceToggle = $('#human-comparison-source');
    var noteEl = $('#human-baseline-note');
    var baselineSection = $('#human-baseline-highlights');
    var baselineBestYear = $('#baseline-best-year');
    var baselineBestYearNote = $('#baseline-best-year-note');
    var baselineBestGrade = $('#baseline-best-grade');
    var baselineBestGradeNote = $('#baseline-best-grade-note');

    var runView = $('#human-run-view');
    var cohortView = $('#human-cohort-view');
    var chartsSection = $('#human-charts');

    var runTableBody = $('#human-run-table-body');
    var cohortTableBody = $('#human-cohort-table-body');

    var runPercentile = $('#human-run-percentile');
    var runPercentileNote = $('#human-run-percentile-note');
    var runZScore = $('#human-run-zscore');
    var runZScoreNote = $('#human-run-zscore-note');

    var statsTotal = $('#stats-total');
    var statsLlmWins = $('#stats-llm-wins');
    var statsHumanWins = $('#stats-human-wins');
    var statsAvgPercentile = $('#stats-avg-percentile');
    var statsBestYear = $('#stats-best-year');
    var statsBestGrade = $('#stats-best-grade');

    var statsThresholds = $('#stats-thresholds');
    var runToplists = $('#human-run-toplists');
    var topLlmWinsList = $('#top-llm-wins');
    var topHumanWinsList = $('#top-human-wins');

    var cdfCanvas = $('#human-cdf-chart');
    var heatmapCanvas = $('#human-heatmap-chart');
    var percentileCanvas = $('#human-percentile-chart');

    function labelForRun(run){
      if (!run) return '';
      return (run.model_label || run.model_id || run.run_id);
    }

    var runTable = document.getElementById('human-run-table');
    var runTableHeaders = runTable ? $$('#human-run-table thead th[data-sort-key]') : [];

    function updateRunTableSortIndicators(){
      if (!runTableHeaders || !runTableHeaders.length) return;
      runTableHeaders.forEach(function(th){
        if (!th) return;
        var sortKey = th.dataset.sortKey;
        if (!sortKey) return;
        var isActive = state.runTableSortOverride && state.runTableSortOverride.column === sortKey;
        if (isActive){
          var direction = state.runTableSortOverride.direction === 'desc' ? 'descending' : 'ascending';
          th.setAttribute('aria-sort', direction);
          th.classList.add('is-sorted');
        } else {
          th.setAttribute('aria-sort', 'none');
          th.classList.remove('is-sorted');
        }
      });
    }

    function setSelectedOption(select, value){
      if (!select) return;
      var has = false;
      Array.from(select.options).forEach(function(option){
        if (option.value === value){
          option.selected = true;
          has = true;
        } else if (!select.multiple){
          option.selected = false;
        }
      });
      if (!has && select.options.length){
        select.options[0].selected = true;
      }
    }

    function populateRunSelectors(){
      if (runSelect){
        runSelect.innerHTML = '';
      }
      if (runChecklist){
        runChecklist.innerHTML = '';
      }
      state.runs.forEach(function(run){
        var label = labelForRun(run) + ' · ' + run.run_id;
        if (runSelect){
          var opt = document.createElement('option');
          opt.value = run.run_id;
          opt.textContent = label;
          runSelect.appendChild(opt);
        }
        if (runChecklist){
          var labelEl = document.createElement('label');
          labelEl.className = 'checkbox-label';
          var cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.value = run.run_id;
          cb.addEventListener('change', function(){
            state.selectedRuns = getSelectedRuns();
            renderCohortView();
          });
          var span = document.createElement('span');
          span.textContent = label;
          labelEl.appendChild(cb);
          labelEl.appendChild(span);
          runChecklist.appendChild(labelEl);
        }
      });
      if (!state.selectedRun && state.runs.length){
        state.selectedRun = state.runs[0].run_id;
      }
      if (runSelect && state.selectedRun){
        setSelectedOption(runSelect, state.selectedRun);
      }
    }

    function updateGroupingMode(){
      var mode = state.selectedGroupingMode || 'year';
      if (mode === 'year'){
        if (yearSelectWrapper) yearSelectWrapper.style.order = '1';
        if (gradeSelectWrapper) gradeSelectWrapper.style.order = '2';
        populateYearSelect();
        if (state.selectedYear){
          populateGradeSelect(state.selectedYear);
        }
      } else {
        if (gradeSelectWrapper) gradeSelectWrapper.style.order = '1';
        if (yearSelectWrapper) yearSelectWrapper.style.order = '2';
        populateAllGradeOptions();
        if (state.selectedGrade){
          populateYearsForGrade(state.selectedGrade);
        }
      }
    }

    function toGradeNumber(value){
      var num = parseInt(value, 10);
      return isNaN(num) ? null : num;
    }

    function parseGradeKeyMembers(key){
      if (!key) return [];
      var text = String(key).trim();
      if (!text) return [];
      text = text
        .replace(/^_range_aggregated_/, '')
        .replace(/_(best|average)$/i, '')
        .replace(/[\s]/g, '')
        .replace(/[\/]/g, '-')
        .replace(/\u2013/g, '-');
      var rangeMatch = text.match(/^(-?\d+)-(-?\d+)$/);
      if (rangeMatch){
        var start = parseInt(rangeMatch[1], 10);
        var end = parseInt(rangeMatch[2], 10);
        if (!isNaN(start) && !isNaN(end)){
          var members = [];
          if (start <= end){
            for (var i = start; i <= end; i++) members.push(i);
          } else {
            for (var j = start; j >= end; j--) members.push(j);
          }
          return members;
        }
      }
      var single = parseInt(text, 10);
      if (!isNaN(single)){
        return [single];
      }
      var matches = text.match(/\d+/g);
      if (!matches) return [];
      if (matches.length >= 2){
        var first = parseInt(matches[0], 10);
        var second = parseInt(matches[1], 10);
        if (!isNaN(first) && !isNaN(second)){
          var nums = [];
          if (first <= second){
            for (var k = first; k <= second; k++) nums.push(k);
          } else {
            for (var m = first; m >= second; m--) nums.push(m);
          }
          return nums;
        }
      }
      var unique = new Set();
      matches.forEach(function(token){
        var value = parseInt(token, 10);
        if (!isNaN(value)){
          unique.add(value);
        }
      });
      return Array.from(unique).sort(function(a, b){ return a - b; });
    }

    function getEntryMembers(entry){
      if (!entry) return [];
      if (Array.isArray(entry.members) && entry.members.length){
        return entry.members
          .map(function(value){ return Number(value); })
          .filter(function(num){ return !isNaN(num); })
          .sort(function(a, b){ return a - b; });
      }
      return parseGradeKeyMembers(entry.grade_label || entry.grade_id);
    }

    function entryHasGrade(entry, gradeValue){
      var gradeNumber = toGradeNumber(gradeValue);
      if (gradeNumber === null) return false;
      var members = getEntryMembers(entry);
      return members.some(function(member){ return member === gradeNumber; });
    }

    function buildGradeDisplay(entry, gradeValue, year){
      if (!entry){
        return { primary: '–', meta: null };
      }
      var members = getEntryMembers(entry);
      var metaParts = [];

      if (gradeValue !== null && entryHasGrade(entry, gradeValue)){
        var primary = 'Grade ' + gradeValue;
        var sourceLabel = entry.grade_label || entry.grade_id;

        var yearInfo = year ? state.yearMap[year] : null;
        var directGradeLabel = null;
        if (yearInfo && Array.isArray(yearInfo.grades)){
          var directGrade = yearInfo.grades.find(function(g){ return g && g.id === String(gradeValue); });
          if (directGrade && directGrade.label){
            directGradeLabel = directGrade.label;
          }
        }

        if (directGradeLabel){
          metaParts.push('Human baseline: Grade ' + directGradeLabel);
        } else if (sourceLabel){
          var cleaned = String(sourceLabel).split('(')[0].trim();
          if (cleaned){
            metaParts.push('Human baseline: ' + cleaned);
          }
        }

        if (!directGradeLabel && entry.grade_id && entry.grade_id.startsWith('_range_aggregated_')){
          metaParts.push('Strategy: ' + (state.aggregationStrategy === 'best' ? 'best-of-range' : 'average-of-range'));
        }
        if (!directGradeLabel && members.length > 1){
          metaParts.push('Covers ' + members.join('-'));
        }
        return { primary: primary, meta: metaParts.length ? metaParts.join(' • ') : null };
      }

      var fallbackLabel = entry.grade_label || entry.grade_id || '–';
      if (members.length > 1){
        metaParts.push('Covers ' + members.join('-'));
      }
      return { primary: fallbackLabel, meta: metaParts.length ? metaParts.join(' • ') : null };
    }

    function deriveGradeMetrics(entry, gradeValue){
      var gradeKey = gradeValue !== null ? String(gradeValue) : null;
      var overrides = entry && entry.member_overrides ? entry.member_overrides : null;
      var override = gradeKey && overrides && overrides[gradeKey] ? overrides[gradeKey] : null;

      var humanMean = override && override.human_mean !== null && override.human_mean !== undefined
        ? override.human_mean
        : entry ? entry.human_mean : null;
      var humanBest = override && override.human_best !== null && override.human_best !== undefined
        ? override.human_best
        : entry ? entry.human_best : null;
      var humanStd = override && override.human_std !== null && override.human_std !== undefined
        ? override.human_std
        : entry ? entry.human_std : null;
      var humanPercentile = override && override.human_percentile !== null && override.human_percentile !== undefined
        ? override.human_percentile
        : entry ? entry.human_percentile : null;
      var zScore = override && override.z_score !== null && override.z_score !== undefined
        ? override.z_score
        : entry ? entry.z_score : null;
      var maxPoints = override && override.max_points !== null && override.max_points !== undefined
        ? override.max_points
        : entry ? entry.max_points : null;
      var totalCount = override && override.total_count !== null && override.total_count !== undefined
        ? override.total_count
        : null;

      return {
        humanMean: humanMean,
        humanBest: humanBest,
        humanStd: humanStd,
        humanPercentile: humanPercentile,
        zScore: zScore,
        maxPoints: maxPoints,
        totalCount: totalCount,
      };
    }

    function collectGradesFromYear(entry){
      var result = new Set();
      if (!entry) return result;
      if (Array.isArray(entry.grades)){
        entry.grades.forEach(function(grade){
          if (Array.isArray(grade.members)){
            grade.members.forEach(function(member){
              var num = Number(member);
              if (!isNaN(num)) result.add(num);
            });
          }
        });
      }
      if (entry.ui_groups){
        Object.values(entry.ui_groups).forEach(function(groupMembers){
          if (Array.isArray(groupMembers)){
            groupMembers.forEach(function(member){
              var num = Number(member);
              if (!isNaN(num)) result.add(num);
            });
          }
        });
      }
      return result;
    }

    function yearHasGrade(entry, gradeValue){
      var gradeNumber = toGradeNumber(gradeValue);
      if (gradeNumber === null || !entry) return false;
      if (Array.isArray(entry.grades)){
        for (var i = 0; i < entry.grades.length; i++){
          var grade = entry.grades[i];
          if (Array.isArray(grade.members) && grade.members.some(function(member){ return Number(member) === gradeNumber; })){
            return true;
          }
        }
      }
      if (entry.ui_groups){
        var groups = Object.values(entry.ui_groups);
        for (var j = 0; j < groups.length; j++){
          var members = groups[j];
          if (Array.isArray(members) && members.some(function(member){ return Number(member) === gradeNumber; })){
            return true;
          }
        }
      }
      return false;
    }

    function populateYearSelect(){
      if (!yearSelect) return;
      var years = state.years.slice().sort(function(a, b){ return a.year - b.year; });
      yearSelect.innerHTML = '';
      years.forEach(function(entry){
        var option = document.createElement('option');
        option.value = String(entry.year);
        option.textContent = String(entry.year);
        yearSelect.appendChild(option);
      });
      if (!state.selectedYear && years.length){
        state.selectedYear = years[0].year;
      }
      if (state.selectedYear && yearSelect.options.length){
        setSelectedOption(yearSelect, String(state.selectedYear));
      }
    }

    function populateAllGradeOptions(){
      if (!gradeSelect) return;
      gradeSelect.innerHTML = '';
      var aggregateGrades = new Set();
      state.years.forEach(function(yearEntry){
        collectGradesFromYear(yearEntry).forEach(function(value){
          aggregateGrades.add(value);
        });
      });
      var grades = Array.from(aggregateGrades).sort(function(a, b){ return a - b; });
      grades.forEach(function(gradeNumber){
        var option = document.createElement('option');
        option.value = String(gradeNumber);
        option.textContent = 'Grade ' + gradeNumber;
        gradeSelect.appendChild(option);
      });
      if (!state.selectedGrade && grades.length){
        state.selectedGrade = String(grades[0]);
      }
      if (state.selectedGrade && gradeSelect.options.length){
        setSelectedOption(gradeSelect, String(state.selectedGrade));
      }
    }

    function populateYearsForGrade(gradeValue){
      if (!yearSelect) return;
      yearSelect.innerHTML = '';
      var yearsWithGrade = state.years
        .filter(function(yearEntry){ return yearHasGrade(yearEntry, gradeValue); })
        .sort(function(a, b){ return a.year - b.year; });
      yearsWithGrade.forEach(function(entry){
        var option = document.createElement('option');
        option.value = String(entry.year);
        option.textContent = String(entry.year);
        yearSelect.appendChild(option);
      });
      if ((!state.selectedYear || !yearsWithGrade.some(function(entry){ return entry.year === state.selectedYear; })) && yearsWithGrade.length){
        state.selectedYear = yearsWithGrade[0].year;
      }
      if (state.selectedYear && yearSelect.options.length){
        setSelectedOption(yearSelect, String(state.selectedYear));
      }
    }

    function populateGradeSelect(year){
      if (!gradeSelect) return;
      gradeSelect.innerHTML = '';
      var entry = state.yearMap[year];
      var gradesForYear = Array.from(collectGradesFromYear(entry)).sort(function(a, b){ return a - b; });
      gradesForYear.forEach(function(gradeNumber){
        var option = document.createElement('option');
        option.value = String(gradeNumber);
        option.textContent = 'Grade ' + gradeNumber;
        gradeSelect.appendChild(option);
      });
      if (!state.selectedGrade || gradesForYear.indexOf(toGradeNumber(state.selectedGrade)) === -1){
        state.selectedGrade = gradesForYear.length ? String(gradesForYear[0]) : null;
      }
      if (state.selectedGrade){
        setSelectedOption(gradeSelect, String(state.selectedGrade));
      }
    }

    function getSelectedRuns(){
      if (!runChecklist) return [];
      var boxes = runChecklist.querySelectorAll('input[type="checkbox"]');
      return Array.from(boxes).filter(function(cb){ return cb.checked; }).map(function(cb){ return cb.value; });
    }

    function ensureCohortDefaults(){
      if (!state.selectedRuns.length && state.runs.length){
        state.selectedRuns = state.runs.slice(0, Math.min(3, state.runs.length)).map(function(run){ return run.run_id; });
        if (runChecklist){
          var boxes = runChecklist.querySelectorAll('input[type="checkbox"]');
          Array.from(boxes).forEach(function(cb){ cb.checked = state.selectedRuns.indexOf(cb.value) !== -1; });
        }
      }
    }

    function fetchRunComparison(runId){
      var cacheKey = runId + '_' + state.aggregationStrategy;
      if (state.runComparisons[cacheKey]){
        return Promise.resolve(state.runComparisons[cacheKey]);
      }
      var url = '/api/humans/compare/run/' + encodeURIComponent(runId) + '?late_year_strategy=' + encodeURIComponent(state.aggregationStrategy);
      return fetchJSON(url)
        .then(function(payload){
          state.runComparisons[cacheKey] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load human comparison for run', runId, err);
          return null;
        });
    }

    function buildRunSummaryKey(runId){
      return [runId, state.selectedComparisonSource, state.aggregationStrategy].join('|');
    }

    function fetchRunSummary(runId){
      var key = buildRunSummaryKey(runId);
      if (state.runSummaries[key]){
        return Promise.resolve(state.runSummaries[key]);
      }
      var params = {
        comparator: state.selectedComparisonSource,
        weight_mode: 'micro',
        late_year_strategy: state.aggregationStrategy,
        top_limit: 5
      };
      var url = '/api/humans/stats/run/' + encodeURIComponent(runId);
      return fetchJSON(url, params)
        .then(function(payload){
          state.runSummaries[key] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load run summary', err);
          return null;
        });
    }

    function buildCohortSummaryKey(runIds){
      return [runIds.slice().sort().join('|'), state.selectedComparisonSource, state.selectedCohortType, state.aggregationStrategy].join('|');
    }

    function fetchCohortSummary(runIds){
      var key = buildCohortSummaryKey(runIds);
      if (state.cohortSummaries[key]){
        return Promise.resolve(state.cohortSummaries[key]);
      }
      if (!runIds.length){
        return Promise.resolve(null);
      }
      var payload = {
        run_ids: runIds,
        comparator: state.selectedComparisonSource,
        weight_mode: state.selectedCohortType,
        late_year_strategy: state.aggregationStrategy,
        top_limit: 5,
      };
      return fetch('/api/humans/stats/cohort', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
        .then(function(response){
          if (!response.ok){
            throw new Error('Request failed: ' + response.status);
          }
          return response.json();
        })
        .then(function(data){
          state.cohortSummaries[key] = data;
          return data;
        })
        .catch(function(err){
          console.error('Failed to load cohort summary', err);
          return null;
        });
    }

    function fetchHumanBaselineSummary(comparator){
      if (state.baselineSummaries[comparator]){
        return Promise.resolve(state.baselineSummaries[comparator]);
      }
      return fetchJSON('/api/humans/stats/human-baseline', { comparator: comparator })
        .then(function(payload){
          state.baselineSummaries[comparator] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load human baseline summary', err);
          return null;
        });
    }

    function fetchYearSummary(year){
      if (state.yearSummaryCache[year]){
        return Promise.resolve(state.yearSummaryCache[year]);
      }
      return fetchJSON('/api/humans/' + year + '/summary')
        .then(function(payload){
          state.yearSummaryCache[year] = payload;
          return payload;
        });
    }

    function fetchCdf(year, gradeId){
      var key = year + ':' + gradeId;
      if (state.cdfCache[key]){
        return Promise.resolve(state.cdfCache[key]);
      }
      return fetchJSON('/api/humans/' + year + '/cdf', { grade: gradeId })
        .then(function(payload){
          state.cdfCache[key] = payload;
          return payload;
        });
    }

    function fetchPercentile(year, gradeId, score){
      var key = year + ':' + gradeId + ':' + score;
      if (state.percentileCache[key]){
        return Promise.resolve(state.percentileCache[key]);
      }
      return fetchJSON('/api/humans/percentile', { year: year, grade: gradeId, score: score })
        .then(function(payload){
          state.percentileCache[key] = payload;
          return payload;
        });
    }

    function fetchCohort(runIds){
      var key = runIds.slice().sort().join('|');
      if (!key){
        return Promise.resolve(null);
      }
      if (state.cohortCache[key]){
        return Promise.resolve(state.cohortCache[key]);
      }
      return fetch('/api/humans/compare/aggregate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds })
      })
        .then(function(response){
          if (!response.ok){
            throw new Error('Request failed: ' + response.status);
          }
          return response.json();
        })
        .then(function(payload){
          state.cohortCache[key] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load cohort comparison', err);
          return null;
        });
    }

    function formatPercent(value){
      if (value === null || value === undefined || isNaN(value)) return '–';
      return (Number(value) * 100).toFixed(1).replace(/\.0$/, '') + '%';
    }

    function formatScore(value){
      if (value === null || value === undefined || isNaN(value)) return '–';
      return Number(value).toFixed(1).replace(/\.0$/, '');
    }

    function formatSignedPercent(value){
      if (value === null || value === undefined || isNaN(value)) return '–';
      var pct = Number(value) * 100;
      var formatted = pct.toFixed(1).replace(/\.0$/, '');
      if (pct > 0) return '+' + formatted + '%';
      if (pct === 0) return '0%';
      return formatted + '%';
    }

    function describeBestYear(bestYear){
      if (!bestYear) return '–';
      var pieces = [];
      if (bestYear.avg_percentile !== null && bestYear.avg_percentile !== undefined){
        pieces.push(formatPercent(bestYear.avg_percentile));
      } else if (bestYear.avg_score_pct !== null && bestYear.avg_score_pct !== undefined){
        pieces.push(formatPercent(bestYear.avg_score_pct));
      } else if (bestYear.avg_gap_pct !== null && bestYear.avg_gap_pct !== undefined){
        pieces.push(formatSignedPercent(bestYear.avg_gap_pct));
      }
      return String(bestYear.year) + (pieces.length ? ' (' + pieces[0] + ')' : '');
    }

    function describeBestGrade(bestGrade){
      if (!bestGrade) return '–';
      var label = bestGrade.grade_label || bestGrade.grade_id;
      var pieces = [];
      if (bestGrade.avg_percentile !== null && bestGrade.avg_percentile !== undefined){
        pieces.push(formatPercent(bestGrade.avg_percentile));
      } else if (bestGrade.avg_score_pct !== null && bestGrade.avg_score_pct !== undefined){
        pieces.push(formatPercent(bestGrade.avg_score_pct));
      } else if (bestGrade.avg_gap_pct !== null && bestGrade.avg_gap_pct !== undefined){
        pieces.push(formatSignedPercent(bestGrade.avg_gap_pct));
      }
      return label + (pieces.length ? ' (' + pieces[0] + ')' : '');
    }

    function renderThresholds(summary){
      if (!statsThresholds) return;
      statsThresholds.innerHTML = '';
      if (!summary || !summary.threshold_breakdown || !summary.threshold_breakdown.length){
        return;
      }
      summary.threshold_breakdown.forEach(function(entry){
        var item = document.createElement('li');
        var thresholdLabel = '≥' + Number(entry.threshold_pp).toFixed(1).replace(/\.0$/, '') + ' p.p.';
        item.textContent = thresholdLabel + ': LLM ' + entry.llm_win_count + ' · Humans ' + entry.human_win_count;
        statsThresholds.appendChild(item);
      });
    }

    function renderTopLists(summary){
      if (!runToplists) return;
      if (!summary){
        runToplists.hidden = true;
        if (topLlmWinsList) topLlmWinsList.innerHTML = '';
        if (topHumanWinsList) topHumanWinsList.innerHTML = '';
        return;
      }

      var llmEntries = summary.top_llm_wins || [];
      var humanEntries = summary.top_human_wins || [];

      if (topLlmWinsList){
        topLlmWinsList.innerHTML = '';
        if (!llmEntries.length){
          var emptyLlm = document.createElement('li');
          emptyLlm.textContent = 'No strong LLM wins detected';
          topLlmWinsList.appendChild(emptyLlm);
        } else {
          llmEntries.forEach(function(entry){
            topLlmWinsList.appendChild(createTopListItem(entry));
          });
        }
      }

      if (topHumanWinsList){
        topHumanWinsList.innerHTML = '';
        if (!humanEntries.length){
          var emptyHuman = document.createElement('li');
          emptyHuman.textContent = 'No strong human wins detected';
          topHumanWinsList.appendChild(emptyHuman);
        } else {
          humanEntries.forEach(function(entry){
            topHumanWinsList.appendChild(createTopListItem(entry));
          });
        }
      }

      runToplists.hidden = !llmEntries.length && !humanEntries.length;
    }

    function createTopListItem(entry){
      var li = document.createElement('li');
      var label = entry.year + ' · ' + (entry.grade_label || entry.grade_id);
      var gapText = formatSignedPercent(entry.gap_score_pct);
      var llmPct = formatPercent(entry.llm_score_pct);
      var humanPct = formatPercent(entry.human_score_pct);
      li.textContent = label + ': ' + gapText + ' (LLM ' + llmPct + ' vs Human ' + humanPct + ')';
      return li;
    }

    function applyRunSummary(summary){
      state.currentRunSummary = summary;
      if (!summary){
        if (statsTotal) statsTotal.textContent = '0';
        if (statsLlmWins) statsLlmWins.textContent = '0';
        if (statsHumanWins) statsHumanWins.textContent = '0';
        if (statsAvgPercentile) statsAvgPercentile.textContent = '–';
        if (statsBestYear) statsBestYear.textContent = '–';
        if (statsBestGrade) statsBestGrade.textContent = '–';
        renderThresholds(null);
        renderTopLists(null);
        return;
      }

      if (statsTotal) statsTotal.textContent = String(summary.total_cells || 0);
      if (statsLlmWins) statsLlmWins.textContent = String(summary.llm_win_count || 0);
      if (statsHumanWins) statsHumanWins.textContent = String(summary.human_win_count || 0);
      if (statsAvgPercentile) statsAvgPercentile.textContent = summary.avg_percentile !== null && summary.avg_percentile !== undefined ? formatPercent(summary.avg_percentile) : '–';
      if (statsBestYear) statsBestYear.textContent = describeBestYear(summary.best_year);
      if (statsBestGrade) statsBestGrade.textContent = describeBestGrade(summary.best_grade);

      renderThresholds(summary);
      renderTopLists(summary);
    }

    function updateBaselineHighlights(){
      if (!baselineSection) return;
      fetchHumanBaselineSummary(state.selectedComparisonSource).then(function(payload){
        if (!payload){
          baselineSection.hidden = true;
          return;
        }
        baselineSection.hidden = false;
        if (baselineBestYear) baselineBestYear.textContent = describeBestYear(payload.best_year);
        if (baselineBestYearNote) baselineBestYearNote.textContent = payload.best_year && payload.best_year.avg_score_pct !== null && payload.best_year.avg_score_pct !== undefined
          ? 'Avg score: ' + formatPercent(payload.best_year.avg_score_pct)
          : '';
        if (baselineBestGrade) baselineBestGrade.textContent = describeBestGrade(payload.best_grade);
        if (baselineBestGradeNote) baselineBestGradeNote.textContent = payload.best_grade && payload.best_grade.avg_score_pct !== null && payload.best_grade.avg_score_pct !== undefined
          ? 'Avg score: ' + formatPercent(payload.best_grade.avg_score_pct)
          : '';
      });
    }

    function updateRunSummary(entry){
      var runRating = $('#human-run-rating');
      var runRatingNote = $('#human-run-rating-note');
      
      var llmTotalScore = $('#llm-total-score');
      var llmScorePct = $('#llm-score-pct');
      var llmPointsEarned = $('#llm-points-earned');
      var llmMaxPoints = $('#llm-max-points');
      
      var humanAvgScore = $('#human-avg-score');
      var humanAvgPct = $('#human-avg-pct');
      var humanBestScore = $('#human-best-score');
      var humanBestPct = $('#human-best-pct');
      var humanAvgScoreWrapper = $('#human-avg-score-wrapper');
      var humanBestScoreWrapper = $('#human-best-score-wrapper');
      var humanComparisonLabel = $('#human-comparison-label');
      var humanStdDev = $('#human-std-dev');
      var humanTotalCount = $('#human-total-count');
      
      var gapScoreDiff = $('#gap-score-diff');
      var gapScoreNote = $('#gap-score-note');
      var gapPctDiff = $('#gap-pct-diff');
      var gapPctNote = $('#gap-pct-note');
      
      if (!entry){
        runPercentile.textContent = '–';
        runPercentileNote.textContent = '';
        runZScore.textContent = '–';
        runZScoreNote.textContent = '';
        if (runRating) runRating.textContent = '–';
        if (runRatingNote) runRatingNote.textContent = '';
        
        if (llmTotalScore) llmTotalScore.textContent = '–';
        if (llmScorePct) llmScorePct.textContent = '–';
        if (llmPointsEarned) llmPointsEarned.textContent = '–';
        if (llmMaxPoints) llmMaxPoints.textContent = '–';
        
        if (humanAvgScore) humanAvgScore.textContent = '–';
        if (humanAvgPct) humanAvgPct.textContent = '–';
        if (humanBestScore) humanBestScore.textContent = '–';
        if (humanBestPct) humanBestPct.textContent = '–';
        if (humanStdDev) humanStdDev.textContent = '–';
        if (humanTotalCount) humanTotalCount.textContent = '–';
        
        if (gapScoreDiff) gapScoreDiff.textContent = '–';
        if (gapScoreNote) gapScoreNote.textContent = '';
        if (gapPctDiff) gapPctDiff.textContent = '–';
        if (gapPctNote) gapPctNote.textContent = '';
        return;
      }

      var selectedGradeNumber = toGradeNumber(state.selectedGrade);
      var metrics = deriveGradeMetrics(entry, selectedGradeNumber);

      if (metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
        runPercentile.textContent = formatPercent(metrics.humanPercentile);
        var pctValue = metrics.humanPercentile * 100;
        if (pctValue >= 99) runPercentileNote.textContent = 'Exceptional - Top 1%';
        else if (pctValue >= 95) runPercentileNote.textContent = 'Excellent - Top 5%';
        else if (pctValue >= 75) runPercentileNote.textContent = 'Good - Top 25%';
        else if (pctValue >= 50) runPercentileNote.textContent = 'Above average';
        else if (pctValue >= 25) runPercentileNote.textContent = 'Below average';
        else runPercentileNote.textContent = 'Low performance';
      } else {
        runPercentile.textContent = '–';
        runPercentileNote.textContent = 'Human percentile unavailable';
      }
      
      if (metrics.zScore !== null && metrics.zScore !== undefined){
        runZScore.textContent = metrics.zScore.toFixed(2);
        var zVal = metrics.zScore;
        if (zVal >= 3.0) runZScoreNote.textContent = '+3σ - Exceptional';
        else if (zVal >= 2.0) runZScoreNote.textContent = '+2σ - Excellent';
        else if (zVal >= 1.0) runZScoreNote.textContent = '+1σ - Good';
        else if (zVal >= 0) runZScoreNote.textContent = 'Above mean';
        else if (zVal >= -1.0) runZScoreNote.textContent = 'Below mean';
        else runZScoreNote.textContent = 'Well below mean';
      } else {
        runZScore.textContent = '–';
        runZScoreNote.textContent = 'Human variance unavailable';
      }
      
      
      if (runRating && runRatingNote){
        if (metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          var percentile = metrics.humanPercentile;
          if (percentile >= 0.99){
            runRating.textContent = '⭐⭐⭐⭐⭐';
            runRatingNote.textContent = 'Exceptional';
          } else if (percentile >= 0.95){
            runRating.textContent = '⭐⭐⭐⭐';
            runRatingNote.textContent = 'Excellent';
          } else if (percentile >= 0.75){
            runRating.textContent = '⭐⭐⭐';
            runRatingNote.textContent = 'Good';
          } else if (percentile >= 0.50){
            runRating.textContent = '⭐⭐';
            runRatingNote.textContent = 'Average';
          } else {
            runRating.textContent = '⭐';
            runRatingNote.textContent = 'Below average';
          }
        } else {
          runRating.textContent = '–';
          runRatingNote.textContent = '';
        }
      }
      
      if (llmTotalScore) llmTotalScore.textContent = entry.llm_total !== null ? formatScore(entry.llm_total) : '–';
      if (llmScorePct) llmScorePct.textContent = entry.llm_score_pct !== null ? formatPercent(entry.llm_score_pct) : '–';
      if (llmPointsEarned) llmPointsEarned.textContent = entry.llm_points_awarded !== null ? formatScore(entry.llm_points_awarded) : '–';
      if (llmMaxPoints) llmMaxPoints.textContent = entry.llm_max !== null ? formatScore(entry.llm_max) : '–';
      
      if (humanComparisonLabel) {
        humanComparisonLabel.textContent = state.selectedComparisonSource === 'best' ? 'Human Best' : 'Human Average';
      }
      if (humanAvgScoreWrapper) {
        humanAvgScoreWrapper.style.display = state.selectedComparisonSource === 'average' ? '' : 'none';
      }
      if (humanBestScoreWrapper) {
        humanBestScoreWrapper.style.display = state.selectedComparisonSource === 'best' ? '' : 'none';
      }

      if (humanAvgScore) humanAvgScore.textContent = metrics.humanMean !== null && metrics.humanMean !== undefined ? formatScore(metrics.humanMean) : '–';
      if (humanBestScore) humanBestScore.textContent = metrics.humanBest !== null && metrics.humanBest !== undefined ? formatScore(metrics.humanBest) : '–';
      if (humanStdDev) humanStdDev.textContent = metrics.humanStd !== null && metrics.humanStd !== undefined ? formatScore(metrics.humanStd) : '–';
      
      var maxPointsForPct = metrics.maxPoints !== null && metrics.maxPoints !== undefined ? metrics.maxPoints : entry.llm_max;
      if (humanAvgPct && metrics.humanMean !== null && metrics.humanMean !== undefined && maxPointsForPct !== null && maxPointsForPct > 0){
        var humanAvgPctVal = metrics.humanMean / maxPointsForPct;
        humanAvgPct.textContent = formatPercent(humanAvgPctVal);
      } else if (humanAvgPct){
        humanAvgPct.textContent = '–';
      }

      if (humanBestPct && metrics.humanBest !== null && metrics.humanBest !== undefined && maxPointsForPct !== null && maxPointsForPct > 0){
        var humanBestPctVal = metrics.humanBest / maxPointsForPct;
        humanBestPct.textContent = formatPercent(humanBestPctVal);
      } else if (humanBestPct){
        humanBestPct.textContent = '–';
      }
      
      if (humanTotalCount){
        if (metrics.totalCount !== null && metrics.totalCount !== undefined){
          humanTotalCount.textContent = metrics.totalCount.toLocaleString();
        } else {
          humanTotalCount.textContent = '–';
        }
      }
      
      var humanScoreForGap = state.selectedComparisonSource === 'best' ? metrics.humanBest : metrics.humanMean;
      var humanScoreLabelForGap = state.selectedComparisonSource === 'best' ? 'best' : 'average';

      if (gapScoreDiff && entry.llm_total !== null && humanScoreForGap !== null){
        var scoreDiff = entry.llm_total - humanScoreForGap;
        gapScoreDiff.textContent = (scoreDiff >= 0 ? '+' : '') + formatScore(scoreDiff);
        gapScoreDiff.className = 'gap-value ' + (scoreDiff >= 0 ? 'positive' : 'negative');
        if (gapScoreNote){
          if (scoreDiff >= 0){
            gapScoreNote.textContent = 'LLM outperforms human ' + humanScoreLabelForGap + ' by ' + formatScore(Math.abs(scoreDiff)) + ' points';
          } else {
            gapScoreNote.textContent = 'Human ' + humanScoreLabelForGap + ' outperforms LLM by ' + formatScore(Math.abs(scoreDiff)) + ' points';
          }
        }
      } else if (gapScoreDiff){
        gapScoreDiff.textContent = '–';
        gapScoreDiff.className = 'gap-value';
      }
      
      if (gapPctDiff && entry.llm_score_pct !== null && humanScoreForGap !== null && maxPointsForPct !== null && maxPointsForPct > 0){
        var humanPctVal = humanScoreForGap / maxPointsForPct;
        var pctDiff = entry.llm_score_pct - humanPctVal;
        gapPctDiff.textContent = (pctDiff >= 0 ? '+' : '') + formatPercent(pctDiff);
        if (gapPctNote){
          var pctPoints = pctDiff * 100;
          if (pctDiff >= 0){
            gapPctNote.textContent = 'LLM ahead by ' + Math.abs(pctPoints).toFixed(1) + ' p.p.';
          } else {
            gapPctNote.textContent = 'Human ' + humanScoreLabelForGap + ' ahead by ' + Math.abs(pctPoints).toFixed(1) + ' p.p.';
          }
        }
      } else if (gapPctDiff){
        gapPctDiff.textContent = '–';
      }
    }

    function renderRunTable(runData){
      if (!runTableBody) return;
      runTableBody.innerHTML = '';
      if (!runData || !runData.entries){
        updateRunTableSortIndicators();
        return;
      }

      var header = $('#human-score-col-header');
      var headerPct = $('#human-score-pct-col-header');
      if (header) {
        header.textContent = state.selectedComparisonSource === 'best' ? 'Best' : 'Avg (μ)';
      }
      if (headerPct) {
        headerPct.textContent = state.selectedComparisonSource === 'best' ? 'Best %' : 'Avg %';
      }

      var filteredEntries = runData.entries;
      var tableView = state.selectedTableView || 'single';
      var selectedGradeNumber = toGradeNumber(state.selectedGrade);

      if (tableView === 'single' && state.selectedYear && selectedGradeNumber !== null){
        filteredEntries = runData.entries.filter(function(entry){
          return entry.year === state.selectedYear && entryHasGrade(entry, selectedGradeNumber);
        });
      } else if (tableView === 'all-years' && selectedGradeNumber !== null){
        filteredEntries = runData.entries.filter(function(entry){
          return entryHasGrade(entry, selectedGradeNumber);
        });
      } else if (tableView === 'all-grades' && state.selectedYear){
        filteredEntries = runData.entries.filter(function(entry){
          return entry.year === state.selectedYear;
        });
      }
      
      if (filteredEntries.length === 0){
        filteredEntries = runData.entries;
      }
      
      var filterSelect = $('#human-run-filter');
      var filterValue = filterSelect ? filterSelect.value : 'all';
      
      if (filterValue !== 'all'){
        filteredEntries = filteredEntries.filter(function(entry){
          var metrics = deriveGradeMetrics(entry, tableView === 'all-grades' ? null : selectedGradeNumber);
          var humanScore = state.selectedComparisonSource === 'best' ? metrics.humanBest : metrics.humanMean;
          if (filterValue === 'excellent') return metrics.humanPercentile >= 0.95;
          if (filterValue === 'good') return metrics.humanPercentile >= 0.75;
          if (filterValue === 'average') return metrics.humanPercentile >= 0.50;
          if (filterValue === 'below-avg') return metrics.humanPercentile < 0.50;
          if (filterValue === 'llm-wins') return entry.llm_total > humanScore;
          if (filterValue === 'human-wins') return entry.llm_total < humanScore;
          return true;
        });
      }
      
      var sortSelect = $('#human-run-sort');
      var sortValue = sortSelect ? sortSelect.value : 'year-grade';
      var gradeNumberForMetrics = tableView === 'all-grades' ? null : selectedGradeNumber;
      var sortOverride = state.runTableSortOverride;
      
      function numericGradeKey(entry){
        if (Array.isArray(entry.members) && entry.members.length){
          return Math.min.apply(null, entry.members);
        }
        var first = String(entry.grade_id || '').split(/[\/-]/)[0];
        var n = parseInt(first, 10);
        return isNaN(n) ? 0 : n;
      }
      
      function compareValues(aVal, bVal, direction){
        var aNull = (aVal === null || aVal === undefined || (typeof aVal === 'number' && isNaN(aVal)));
        var bNull = (bVal === null || bVal === undefined || (typeof bVal === 'number' && isNaN(bVal)));
        if (aNull && bNull) return 0;
        if (aNull) return direction > 0 ? 1 : -1;
        if (bNull) return direction > 0 ? -1 : 1;
        if (typeof aVal === 'string' || typeof bVal === 'string'){
          var result = String(aVal).localeCompare(String(bVal), undefined, { sensitivity: 'base' });
          if (result < 0) return -1 * direction;
          if (result > 0) return 1 * direction;
          return 0;
        }
        var numA = Number(aVal);
        var numB = Number(bVal);
        if (numA < numB) return -1 * direction;
        if (numA > numB) return 1 * direction;
        return 0;
      }

      function fallbackCompare(a, b){
        if (a.year !== b.year) return a.year - b.year;
        return numericGradeKey(a) - numericGradeKey(b);
      }

      function maxPointsFor(entry, metrics){
        if (metrics.maxPoints !== null && metrics.maxPoints !== undefined) return metrics.maxPoints;
        if (entry.llm_max !== null && entry.llm_max !== undefined) return entry.llm_max;
        return null;
      }

      var comparator;
      if (sortOverride && sortOverride.column){
        var sortDirection = sortOverride.direction === 'desc' ? -1 : 1;
        comparator = function(a, b){
          var metricsA = deriveGradeMetrics(a, gradeNumberForMetrics);
          var metricsB = deriveGradeMetrics(b, gradeNumberForMetrics);
          var humanScoreA = state.selectedComparisonSource === 'best' ? metricsA.humanBest : metricsA.humanMean;
          var humanScoreB = state.selectedComparisonSource === 'best' ? metricsB.humanBest : metricsB.humanMean;
          var valueA = null;
          var valueB = null;
          if (sortOverride.column === 'year'){
            valueA = a.year;
            valueB = b.year;
          } else if (sortOverride.column === 'grade'){
            valueA = numericGradeKey(a);
            valueB = numericGradeKey(b);
          } else if (sortOverride.column === 'llm_score'){
            valueA = a.llm_total;
            valueB = b.llm_total;
          } else if (sortOverride.column === 'llm_pct'){
            valueA = a.llm_score_pct;
            valueB = b.llm_score_pct;
          } else if (sortOverride.column === 'human_score'){
            valueA = humanScoreA;
            valueB = humanScoreB;
          } else if (sortOverride.column === 'human_pct'){
            var maxA = maxPointsFor(a, metricsA);
            var maxB = maxPointsFor(b, metricsB);
            valueA = (humanScoreA !== null && maxA) ? humanScoreA / maxA : null;
            valueB = (humanScoreB !== null && maxB) ? humanScoreB / maxB : null;
          } else if (sortOverride.column === 'human_std'){
            valueA = metricsA.humanStd;
            valueB = metricsB.humanStd;
          } else if (sortOverride.column === 'percentile'){
            valueA = metricsA.humanPercentile;
            valueB = metricsB.humanPercentile;
          } else if (sortOverride.column === 'gap'){
            valueA = (a.llm_total !== null && humanScoreA !== null) ? (a.llm_total - humanScoreA) : null;
            valueB = (b.llm_total !== null && humanScoreB !== null) ? (b.llm_total - humanScoreB) : null;
          }
          var result = compareValues(valueA, valueB, sortDirection);
          if (result !== 0) return result;
          return fallbackCompare(a, b);
        };
      } else {
        comparator = function(a, b){
          var metricsA = deriveGradeMetrics(a, gradeNumberForMetrics);
          var metricsB = deriveGradeMetrics(b, gradeNumberForMetrics);
          var humanScoreA = state.selectedComparisonSource === 'best' ? metricsA.humanBest : metricsA.humanMean;
          var humanScoreB = state.selectedComparisonSource === 'best' ? metricsB.humanBest : metricsB.humanMean;

          if (sortValue === 'year-grade'){
            if (a.year !== b.year) return a.year - b.year;
            return numericGradeKey(a) - numericGradeKey(b);
          } else if (sortValue === 'grade-year'){
            var gA = numericGradeKey(a);
            var gB = numericGradeKey(b);
            if (gA !== gB) return gA - gB;
            return (a.year || 0) - (b.year || 0);
          } else if (sortValue === 'percentile-desc'){
            var pa = (metricsA.humanPercentile == null ? -Infinity : metricsA.humanPercentile);
            var pb = (metricsB.humanPercentile == null ? -Infinity : metricsB.humanPercentile);
            if (pb !== pa) return pb - pa;
            if (a.year !== b.year) return a.year - b.year;
            return numericGradeKey(a) - numericGradeKey(b);
          } else if (sortValue === 'percentile-asc'){
            var pa2 = (metricsA.humanPercentile == null ? Infinity : metricsA.humanPercentile);
            var pb2 = (metricsB.humanPercentile == null ? Infinity : metricsB.humanPercentile);
            if (pa2 !== pb2) return pa2 - pb2;
            if (a.year !== b.year) return a.year - b.year;
            return numericGradeKey(a) - numericGradeKey(b);
          } else if (sortValue === 'gap-desc'){
            var gapA = (a.llm_total || 0) - (humanScoreA || 0);
            var gapB = (b.llm_total || 0) - (humanScoreB || 0);
            return gapB - gapA;
          } else if (sortValue === 'gap-asc'){
            var gapA2 = (a.llm_total || 0) - (humanScoreA || 0);
            var gapB2 = (b.llm_total || 0) - (humanScoreB || 0);
            return gapA2 - gapB2;
          } else if (sortValue === 'llm-score-desc'){
            if ((b.llm_total || 0) !== (a.llm_total || 0)) return (b.llm_total || 0) - (a.llm_total || 0);
            if (a.year !== b.year) return a.year - b.year;
            return numericGradeKey(a) - numericGradeKey(b);
          } else if (sortValue === 'human-score-desc'){
            if ((humanScoreB || 0) !== (humanScoreA || 0)) return (humanScoreB || 0) - (humanScoreA || 0);
            if (a.year !== b.year) return a.year - b.year;
            return numericGradeKey(a) - numericGradeKey(b);
          }
          return 0;
        };
      }

      filteredEntries = filteredEntries.slice().sort(comparator);
      updateRunTableSortIndicators();
      
      var bandKey = function(entry){
        if (sortOverride && sortOverride.column) return '';
        if (sortValue === 'year-grade') return String(entry.year);
        if (sortValue === 'grade-year'){
          var key = (Array.isArray(entry.members) && entry.members.length)
            ? Math.min.apply(null, entry.members)
            : (function(){
                var first = String(entry.grade_id || '').split(/[\/-]/)[0];
                var n = parseInt(first, 10);
                return isNaN(n) ? 0 : n;
              })();
          return 'g-' + key;
        }
        return '';
      };

      var currentBand = null;
      var isOddBand = false;
      filteredEntries.forEach(function(entry){
        var tr = document.createElement('tr');
        var thisBand = bandKey(entry);
        if (thisBand !== currentBand){
          currentBand = thisBand;
          isOddBand = !isOddBand;
          tr.classList.add('band-start');
        }
        if (currentBand) tr.classList.add(isOddBand ? 'band-odd' : 'band-even');
        var metrics = deriveGradeMetrics(entry, tableView === 'all-grades' ? null : selectedGradeNumber);
        var percentileClass = '';
        if (metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          if (metrics.humanPercentile >= 0.95) percentileClass = 'excellent';
          else if (metrics.humanPercentile >= 0.75) percentileClass = 'good';
          else if (metrics.humanPercentile >= 0.50) percentileClass = 'average';
          else percentileClass = 'below-average';
        }
        
        var humanScore = state.selectedComparisonSource === 'best' ? metrics.humanBest : metrics.humanMean;
        var humanPct = '–';
        var maxPointsForPct = metrics.maxPoints !== null && metrics.maxPoints !== undefined ? metrics.maxPoints : entry.llm_max;
        if (humanScore !== null && maxPointsForPct !== null && maxPointsForPct > 0){
          humanPct = formatPercent(humanScore / maxPointsForPct);
        }
        
        var gap = '–';
        var gapClass = '';
        if (entry.llm_total !== null && humanScore !== null){
          var scoreDiff = entry.llm_total - humanScore;
          gap = (scoreDiff >= 0 ? '+' : '') + formatScore(scoreDiff);
          gapClass = scoreDiff >= 0 ? 'positive' : 'negative';
        }

        var gradeInfo = buildGradeDisplay(entry, (tableView === 'all-grades' ? null : selectedGradeNumber), entry.year);
        var gradeCell = '<div class="grade-primary">' + gradeInfo.primary + '</div>';
        if (gradeInfo.meta){
          gradeCell += '<div class="grade-meta">' + gradeInfo.meta + '</div>';
        }

        tr.innerHTML = [
          '<td>' + entry.year + '</td>',
          '<td>' + gradeCell + '</td>',
          '<td>' + formatScore(entry.llm_total) + '</td>',
          '<td>' + formatPercent(entry.llm_score_pct) + '</td>',
          '<td>' + (humanScore !== null ? formatScore(humanScore) : '–') + '</td>',
          '<td>' + humanPct + '</td>',
          '<td>' + (metrics.humanStd !== null && metrics.humanStd !== undefined ? formatScore(metrics.humanStd) : '–') + '</td>',
          '<td class="' + percentileClass + '">' + formatPercent(metrics.humanPercentile) + '</td>',
          '<td class="' + gapClass + '">' + gap + '</td>'
        ].join('');
        runTableBody.appendChild(tr);
      });
      
      var aggregatedNoteEl = $('#human-run-aggregated-note');
      if (aggregatedNoteEl){
        var aggregatedEntry = null;
        if (selectedGradeNumber !== null && tableView !== 'all-grades'){
          aggregatedEntry = filteredEntries.find(function(entry){
            return entryHasGrade(entry, selectedGradeNumber) && getEntryMembers(entry).length > 1;
          });
        }
        if (aggregatedEntry){
          var bandLabelRaw = aggregatedEntry.grade_label || aggregatedEntry.grade_id || '';
          var bandLabel = bandLabelRaw.split('(')[0].trim() || bandLabelRaw || 'original band';
          var coverage = getEntryMembers(aggregatedEntry).join('-');
          var coverageText = coverage ? ' covering grades ' + coverage : '';
          aggregatedNoteEl.textContent = 'Note: Contest data only reports this segment as ' + bandLabel + coverageText + '; the table reuses that band for Grade ' + selectedGradeNumber + '.';
          aggregatedNoteEl.hidden = false;
        } else {
          aggregatedNoteEl.hidden = true;
        }
      }

      updateRunTableStats(filteredEntries);
    }
    
    function updateRunTableStats(entries){
      if (state.currentRunSummary){
        applyRunSummary(state.currentRunSummary);
        return;
      }

      var statsTotal = $('#stats-total');
      var statsLlmWins = $('#stats-llm-wins');
      var statsHumanWins = $('#stats-human-wins');
      var statsAvgPercentile = $('#stats-avg-percentile');
      var statsBestYear = $('#stats-best-year');
      var statsBestGrade = $('#stats-best-grade');

      var selectedGradeNumber = (state.selectedTableView === 'all-grades') ? null : toGradeNumber(state.selectedGrade);

      if (!entries || entries.length === 0){
        if (statsTotal) statsTotal.textContent = '0';
        if (statsLlmWins) statsLlmWins.textContent = '0';
        if (statsHumanWins) statsHumanWins.textContent = '0';
        if (statsAvgPercentile) statsAvgPercentile.textContent = '–';
        if (statsBestYear) statsBestYear.textContent = '–';
        if (statsBestGrade) statsBestGrade.textContent = '–';
        return;
      }
      
      var llmWins = 0;
      var humanWins = 0;
      var totalPercentile = 0;
      var percentileCount = 0;
      var yearScores = {};
      var gradeScores = {};
      
      entries.forEach(function(entry){
        var metrics = deriveGradeMetrics(entry, selectedGradeNumber);
        var humanScore = state.selectedComparisonSource === 'best' ? metrics.humanBest : metrics.humanMean;
        if (entry.llm_total !== null && humanScore !== null){
          if (entry.llm_total > humanScore) llmWins++;
          else if (entry.llm_total < humanScore) humanWins++;
        }
        
        if (metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          totalPercentile += metrics.humanPercentile;
          percentileCount++;
        }
        
        if (entry.year && metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          if (!yearScores[entry.year]) yearScores[entry.year] = {sum: 0, count: 0};
          yearScores[entry.year].sum += metrics.humanPercentile;
          yearScores[entry.year].count++;
        }
        
        if (state.selectedGrade && metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          var key = 'Grade ' + state.selectedGrade;
          if (!gradeScores[key]) gradeScores[key] = {sum: 0, count: 0};
          gradeScores[key].sum += metrics.humanPercentile;
          gradeScores[key].count++;
        }
      });
      
      if (statsTotal) statsTotal.textContent = entries.length;
      if (statsLlmWins) statsLlmWins.textContent = llmWins;
      if (statsHumanWins) statsHumanWins.textContent = humanWins;
      if (statsAvgPercentile && percentileCount > 0){
        var avgPct = totalPercentile / percentileCount;
        statsAvgPercentile.textContent = formatPercent(avgPct);
      } else if (statsAvgPercentile){
        statsAvgPercentile.textContent = '–';
      }
      
      var bestYear = null;
      var bestYearScore = -1;
      Object.keys(yearScores).forEach(function(year){
        var avg = yearScores[year].sum / yearScores[year].count;
        if (avg > bestYearScore){
          bestYearScore = avg;
          bestYear = year;
        }
      });
      if (statsBestYear && bestYear){
        statsBestYear.textContent = bestYear + ' (' + formatPercent(bestYearScore) + ')';
      } else if (statsBestYear){
        statsBestYear.textContent = '–';
      }
      
      var bestGrade = null;
      var bestGradeScore = -1;
      Object.keys(gradeScores).forEach(function(grade){
        var avg = gradeScores[grade].sum / gradeScores[grade].count;
        if (avg > bestGradeScore){
          bestGradeScore = avg;
          bestGrade = grade;
        }
      });
      if (statsBestGrade && bestGrade){
        statsBestGrade.textContent = bestGrade + ' (' + formatPercent(bestGradeScore) + ')';
      } else if (statsBestGrade){
        statsBestGrade.textContent = '–';
      }
    }

    function buildHeatmapData(entries, label){
      if (!entries || !entries.length) return null;
      var entry = entries[0];
      var labels = entry.bin_comparison.map(function(item){ return item.bin_id; });
      var values = entry.bin_comparison.map(function(item, idx){
        return [idx, 0, Number(item.delta_smoothed || item.delta || 0) * 100];
      });
      return {
        xLabels: labels,
        yLabels: [label],
        values: values,
      };
    }

    function updateRunCharts(runData, entry){
      if (!entry){
        charts.updateCDF(cdfCanvas, [], [], 0);
        charts.updateHeatmap(heatmapCanvas, { xLabels: [], yLabels: [], values: [] });
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = true;
        return;
      }
      var year = entry.year;
      var gradeId = entry.grade_id;
      Promise.all([
        fetchYearSummary(year),
        fetchCdf(year, gradeId),
        (entry.human_percentile === null || entry.human_percentile === undefined)
          ? fetchPercentile(year, gradeId, entry.llm_total)
          : Promise.resolve(null)
      ]).then(function(results){
        var yearSummary = results[0];
        var cdf = results[1];
        var percentileOverride = results[2];
        if (percentileOverride && percentileOverride.percentile !== undefined){
          entry.human_percentile = percentileOverride.percentile;
        }

        var metrics = deriveGradeMetrics(entry, toGradeNumber(state.selectedGrade));
        var markers = [];
        if (metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
          markers.push({
            label: labelForRun(state.runs.find(function(run){ return run.run_id === runData.run_id; })),
            score: entry.llm_total,
            percentile: metrics.humanPercentile,
          });
        }

        var runGradeInfo = buildGradeDisplay(entry, toGradeNumber(state.selectedGrade), entry.year);
        var runGradeLabel = runGradeInfo.meta ? runGradeInfo.primary + ' • ' + runGradeInfo.meta : runGradeInfo.primary;

        charts.updateCDF(
          cdfCanvas,
          (cdf && cdf.points) ? cdf.points.map(function(point){
            return { score: point.score, percentile: point.percentile };
          }) : [],
          markers,
          yearSummary && yearSummary.grades ? entry.max_points : entry.llm_max
        );

        var heatmapData = buildHeatmapData([entry], runGradeLabel);
        charts.updateHeatmap(heatmapCanvas, heatmapData);
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = false;
      }).catch(function(err){
        console.error('Failed to update run charts', err);
        chartsSection.hidden = false;
      });
    }

    function renderRunView(){
      if (!state.selectedRun){
        runView.hidden = true;
        return;
      }
      runView.hidden = false;
      var cacheKey = state.selectedRun + '_' + state.aggregationStrategy;
      var comparison = state.runComparisons[cacheKey];
      if (!comparison && state.humanComparison && state.humanComparison.run_id === state.selectedRun){
        comparison = state.humanComparison;
        state.runComparisons[cacheKey] = comparison;
      }
      state.runData = comparison || null;
      if (!comparison || !comparison.entries || !comparison.entries.length){
        noteEl.textContent = 'No baseline data available for this run.';
        renderRunTable(null);
        updateRunSummary(null);
        updateRunCharts(null, null);
        return;
      }
      var yearsInRun = comparison.entries.map(function(entry){ return entry.year; });
      if (!state.selectedYear || yearsInRun.indexOf(state.selectedYear) === -1){
        state.selectedYear = yearsInRun[0];
        if (yearSelect){
          setSelectedOption(yearSelect, String(state.selectedYear));
        }
      }
      populateGradeSelect(state.selectedYear);
      var entriesForYear = comparison.entries.filter(function(entry){ return entry.year === state.selectedYear; });
      var selectedGradeNumber = toGradeNumber(state.selectedGrade);
      if (selectedGradeNumber === null || !entriesForYear.some(function(entry){ return entryHasGrade(entry, selectedGradeNumber); })){
        var fallbackEntry = entriesForYear[0];
        var fallbackMembers = getEntryMembers(fallbackEntry);
        state.selectedGrade = fallbackMembers.length ? String(fallbackMembers[0]) : null;
        selectedGradeNumber = toGradeNumber(state.selectedGrade);
        if (gradeSelect && state.selectedGrade){
          setSelectedOption(gradeSelect, String(state.selectedGrade));
        }
      }
      var selectedEntry = entriesForYear.find(function(entry){ return entryHasGrade(entry, selectedGradeNumber); }) || entriesForYear[0];
      if (selectedEntry && selectedGradeNumber === null){
        var membersFallback = getEntryMembers(selectedEntry);
        if (membersFallback.length){
          state.selectedGrade = String(membersFallback[0]);
          selectedGradeNumber = membersFallback[0];
        }
      }
      if (selectedEntry && state.selectedGrade){
        var noteParts = ['Run ' + state.selectedRun];
        var gradeInfo = buildGradeDisplay(selectedEntry, selectedGradeNumber, selectedEntry ? selectedEntry.year : null);
        noteParts.push(gradeInfo.primary);
        if (gradeInfo.meta){
          noteParts.push(gradeInfo.meta);
        }
        noteEl.textContent = noteParts.join(' · ');
      } else if (selectedEntry){
        noteEl.textContent = 'Run ' + state.selectedRun + ' · ' + (selectedEntry.grade_label || selectedEntry.grade_id);
      } else {
        noteEl.textContent = '';
      }
      // quick stats
      updateQuickStats(comparison);

      state.currentRunSummaryKey = buildRunSummaryKey(state.selectedRun);
      applyRunSummary(null);
      fetchRunSummary(state.selectedRun).then(function(payload){
        if (state.currentRunSummaryKey !== buildRunSummaryKey(state.selectedRun)){
          return;
        }
        var summary = payload && payload.summary ? payload.summary : null;
        applyRunSummary(summary);
      });

      renderRunTable(comparison);
      updateRunSummary(selectedEntry);
      updateRunCharts(comparison, selectedEntry);
    }

    function updateQuickStats(runData){
      var elEntries = document.getElementById('quick-entries');
      var elYears = document.getElementById('quick-years');
      var elGrades = document.getElementById('quick-grades');
      var elPoints = document.getElementById('quick-points-earned');
      var elMax = document.getElementById('quick-max-points');
      if (!runData || !runData.entries){
        if (elEntries) elEntries.textContent = '–';
        if (elYears) elYears.textContent = '–';
        if (elGrades) elGrades.textContent = '–';
        if (elPoints) elPoints.textContent = '–';
        if (elMax) elMax.textContent = '–';
        return;
      }
      var entries = runData.entries;
      var years = Array.from(new Set(entries.map(function(e){ return e.year; }))).sort();
      var grades = Array.from(new Set(entries.map(function(e){ return e.grade_label || e.grade_id; })));
      var totalEarned = entries.reduce(function(acc, e){ return acc + (e.llm_points_awarded || 0); }, 0);
      var totalMax = entries.reduce(function(acc, e){ return acc + (e.llm_max || 0); }, 0);
      if (elEntries) elEntries.textContent = String(entries.length);
      if (elYears) elYears.textContent = years[0] + '–' + years[years.length - 1];
      if (elGrades) elGrades.textContent = grades.length + ' groups';
      if (elPoints) elPoints.textContent = totalEarned.toFixed(1).replace(/\.0$/, '');
      if (elMax) elMax.textContent = totalMax.toFixed(1).replace(/\.0$/, '');
    }

    function computeHeatmapValues(entry){
      if (!entry) return null;
      var gradeInfo = buildGradeDisplay(entry, toGradeNumber(state.selectedGrade), entry.year);
      var label = gradeInfo.meta ? gradeInfo.primary + ' • ' + gradeInfo.meta : gradeInfo.primary;
      return buildHeatmapData([entry], label);
    }

    function renderCohortTable(entries){
      if (!cohortTableBody) return;
      cohortTableBody.innerHTML = '';
      if (!entries){
        return;
      }
      var activeGradeNumber = toGradeNumber(state.selectedGrade);
      entries.forEach(function(entry){
        var tr = document.createElement('tr');
        var gradeInfo = buildGradeDisplay(entry, activeGradeNumber !== null && entryHasGrade(entry, activeGradeNumber) ? activeGradeNumber : null, entry.year);
        var gradeCell = '<div class="grade-primary">' + gradeInfo.primary + '</div>';
        if (gradeInfo.meta){
          gradeCell += '<div class="grade-meta">' + gradeInfo.meta + '</div>';
        }
        var metrics = deriveGradeMetrics(entry, activeGradeNumber !== null && entryHasGrade(entry, activeGradeNumber) ? activeGradeNumber : null);
        var avgHumanPercentile = metrics.humanPercentile !== null && metrics.humanPercentile !== undefined
          ? formatPercent(metrics.humanPercentile)
          : formatPercent(entry.avg_human_percentile);
        tr.innerHTML = [
          '<td>' + entry.year + '</td>',
          '<td>' + gradeCell + '</td>',
          '<td>' + entry.run_count + '</td>',
          '<td>' + entry.sample_count + '</td>',
          '<td>' + formatPercent(entry.avg_llm_score_pct) + '</td>',
          '<td>' + avgHumanPercentile + '</td>',
          '<td>' + formatPercent(entry.p25_percentile) + '</td>',
          '<td>' + formatPercent(entry.median_percentile) + '</td>',
          '<td>' + formatPercent(entry.p75_percentile) + '</td>',
          '<td>' + (entry.best_run_id || '–') + '</td>',
          '<td>' + (entry.worst_run_id || '–') + '</td>'
        ].join('');
        cohortTableBody.appendChild(tr);
      });
    }

    function updateCohortCharts(entries, markers){
      if (!entries || !entries.length){
        charts.updateCDF(cdfCanvas, [], [], 0);
        charts.updateHeatmap(heatmapCanvas, { xLabels: [], yLabels: [], values: [] });
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = true;
        return;
      }
      var entry = entries[0];
      var year = entry.year;
      var gradeId = entry.grade_id;
      Promise.all([
        fetchYearSummary(year),
        fetchCdf(year, gradeId)
      ]).then(function(results){
        var cdf = results[1];
        var cohortGradeInfo = buildGradeDisplay(entry, toGradeNumber(state.selectedGrade), entry.year);
        var cohortGradeLabel = cohortGradeInfo.meta ? cohortGradeInfo.primary + ' • ' + cohortGradeInfo.meta : cohortGradeInfo.primary;

        charts.updateCDF(
          cdfCanvas,
          (cdf && cdf.points) ? cdf.points.map(function(point){ return { score: point.score, percentile: point.percentile }; }) : [],
          markers || [],
          entry.max_points
        );
        var heatmapData = buildHeatmapData(entries, cohortGradeLabel);
        charts.updateHeatmap(heatmapCanvas, heatmapData);

        var gradeNumberForBox = toGradeNumber(state.selectedGrade);
        var boxData = [];
        entries.forEach(function(item){
          if (item.median_percentile !== null && item.median_percentile !== undefined){
            var itemInfo = buildGradeDisplay(item, gradeNumberForBox !== null && entryHasGrade(item, gradeNumberForBox) ? gradeNumberForBox : null, item.year);
            var itemLabel = itemInfo.meta ? itemInfo.primary + ' • ' + itemInfo.meta : itemInfo.primary;
            boxData.push({
              label: itemLabel,
              stats: {
                min: item.min_percentile !== null && item.min_percentile !== undefined ? item.min_percentile : item.median_percentile,
                p25: item.p25_percentile !== null && item.p25_percentile !== undefined ? item.p25_percentile : item.median_percentile,
                median: item.median_percentile,
                p75: item.p75_percentile !== null && item.p75_percentile !== undefined ? item.p75_percentile : item.median_percentile,
                max: item.max_percentile !== null && item.max_percentile !== undefined ? item.max_percentile : item.median_percentile,
              }
            });
          }
        });
        charts.updateBoxViolin(percentileCanvas, boxData);
        chartsSection.hidden = false;
      }).catch(function(err){
        console.error('Failed to update cohort charts', err);
        chartsSection.hidden = false;
      });
    }

    function updateCohortSummary(entry, runCount){
      var runCountEl = $('#human-cohort-run-count');
      var avgPercentileEl = $('#human-cohort-avg-percentile');
      var medianPercentileEl = $('#human-cohort-median-percentile');
      var bestRunEl = $('#human-cohort-best-run');
      var rangeEl = $('#human-cohort-range');
      var percentileNoteEl = $('#human-cohort-percentile-note');
      var bestNoteEl = $('#human-cohort-best-note');
      
      if (runCountEl) runCountEl.textContent = runCount || '0';
      
      if (!entry){
        if (avgPercentileEl) avgPercentileEl.textContent = '–';
        if (medianPercentileEl) medianPercentileEl.textContent = '–';
        if (bestRunEl) bestRunEl.textContent = '–';
        if (rangeEl) rangeEl.textContent = '–';
        if (percentileNoteEl) percentileNoteEl.textContent = '';
        if (bestNoteEl) bestNoteEl.textContent = '';
        return;
      }
      
      var gradeNumber = toGradeNumber(state.selectedGrade);
      var metrics = deriveGradeMetrics(entry, gradeNumber);

      if (avgPercentileEl && metrics.humanPercentile !== null && metrics.humanPercentile !== undefined){
        avgPercentileEl.textContent = formatPercent(metrics.humanPercentile);
        if (percentileNoteEl){
          var pct = metrics.humanPercentile * 100;
          if (pct >= 95) percentileNoteEl.textContent = 'Excellent performance';
          else if (pct >= 75) percentileNoteEl.textContent = 'Good performance';
          else if (pct >= 50) percentileNoteEl.textContent = 'Above average';
          else percentileNoteEl.textContent = 'Below average';
        }
      } else if (avgPercentileEl && entry.avg_human_percentile !== null && entry.avg_human_percentile !== undefined){
        avgPercentileEl.textContent = formatPercent(entry.avg_human_percentile);
      } else if (avgPercentileEl){
        avgPercentileEl.textContent = '–';
      }
      
      if (medianPercentileEl && entry.median_percentile !== null && entry.median_percentile !== undefined){
        medianPercentileEl.textContent = formatPercent(entry.median_percentile);
      } else if (medianPercentileEl){
        medianPercentileEl.textContent = '–';
      }
      
      if (bestRunEl && entry.best_run_id){
        var shortId = entry.best_run_id.length > 15 ? '...' + entry.best_run_id.slice(-12) : entry.best_run_id;
        bestRunEl.textContent = shortId;
        if (bestNoteEl){
          var bestPct = entry.max_percentile !== null && entry.max_percentile !== undefined ? 
            formatPercent(entry.max_percentile) : 'N/A';
          bestNoteEl.textContent = 'Percentile: ' + bestPct;
        }
      } else if (bestRunEl){
        bestRunEl.textContent = '–';
      }
      
      if (rangeEl && entry.p25_percentile !== null && entry.p75_percentile !== null){
        rangeEl.textContent = formatPercent(entry.p25_percentile) + ' - ' + formatPercent(entry.p75_percentile);
      } else if (rangeEl){
        rangeEl.textContent = '–';
      }
    }

    function renderCohortView(){
      ensureCohortDefaults();
      var runIds = state.selectedRuns.slice();
      if (!runIds.length){
        cohortView.hidden = true;
        chartsSection.hidden = true;
        return;
      }
      cohortView.hidden = false;
      runMultiWrapper.hidden = false;
      cohortTypeWrapper.hidden = false;
      var key = runIds.slice().sort().join('|');
      fetchCohort(runIds).then(function(payload){
        if (!payload){
          renderCohortTable([]);
          chartsSection.hidden = true;
          return;
        }
        var cohortStats = (state.selectedCohortType === 'macro') ? payload.macro : payload.micro;
        var entries = cohortStats.entries || [];
        if (!entries.length){
          renderCohortTable([]);
          chartsSection.hidden = true;
          return;
        }
        if (!state.selectedYear || !entries.some(function(entry){ return entry.year === state.selectedYear; })){
          state.selectedYear = entries[0].year;
          if (yearSelect){
            setSelectedOption(yearSelect, String(state.selectedYear));
          }
          populateGradeSelect(state.selectedYear);
        }
        var filteredEntries = entries.filter(function(entry){ return entry.year === state.selectedYear; });
        var selectedGradeNumber = toGradeNumber(state.selectedGrade);
        if (selectedGradeNumber === null || !filteredEntries.some(function(entry){ return entryHasGrade(entry, selectedGradeNumber); })){
          var fallback = filteredEntries[0];
          var fallbackMembers = getEntryMembers(fallback);
          state.selectedGrade = fallbackMembers.length ? String(fallbackMembers[0]) : null;
          selectedGradeNumber = toGradeNumber(state.selectedGrade);
          if (gradeSelect && state.selectedGrade){
            setSelectedOption(gradeSelect, String(state.selectedGrade));
          }
        }
        var selectedEntries = filteredEntries.filter(function(entry){ return entryHasGrade(entry, selectedGradeNumber); });
        if (selectedEntries.length){
          var cohortLabelParts = ['Cohort'];
          if (state.selectedGrade){
            var cohortGradeInfo = buildGradeDisplay(selectedEntries[0], selectedGradeNumber, selectedEntries[0] ? selectedEntries[0].year : null);
            cohortLabelParts.push(cohortGradeInfo.primary);
            if (cohortGradeInfo.meta){
              cohortLabelParts.push(cohortGradeInfo.meta);
            }
          } else {
            cohortLabelParts.push(selectedEntries[0].grade_label || selectedEntries[0].grade_id);
          }
          noteEl.textContent = cohortLabelParts.join(' · ');
        } else {
          noteEl.textContent = state.selectedGrade ? ('Cohort · Grade ' + state.selectedGrade) : 'Cohort';
        }
        updateCohortSummary(selectedEntries[0], runIds.length);
        renderCohortTable(filteredEntries);

        Promise.all(runIds.map(fetchRunComparison)).then(function(){
          var markers = [];
          runIds.forEach(function(runId){
            var comparisonKey = runId + '_' + state.aggregationStrategy;
            var comparison = state.runComparisons[comparisonKey];
            if (!comparison || !comparison.entries) return;
            var entry = comparison.entries.find(function(item){
              return item.year === state.selectedYear && entryHasGrade(item, selectedGradeNumber);
            });
            if (entry){
              var entryMetrics = deriveGradeMetrics(entry, selectedGradeNumber);
              if (entryMetrics.humanPercentile !== null && entryMetrics.humanPercentile !== undefined){
                markers.push({
                  label: labelForRun(state.runs.find(function(run){ return run.run_id === runId; })),
                  score: entry.llm_total,
                  percentile: entryMetrics.humanPercentile,
                });
              }
            }
          });
          updateCohortCharts(selectedEntries, markers);
        });
      });
    }

    function updateView(){
      var isRunView = state.selectedView === 'run';
      if (runSelectWrapper) runSelectWrapper.hidden = !isRunView;
      if (runView) runView.hidden = !isRunView;
      if (runMultiWrapper) runMultiWrapper.hidden = isRunView;
      if (cohortTypeWrapper) cohortTypeWrapper.hidden = isRunView;
      if (cohortView) cohortView.hidden = isRunView;
      if (percentileCanvas) percentileCanvas.parentElement.parentElement.parentElement.style.display = isRunView ? 'none' : '';
      if (isRunView){
        if (!state.selectedRun && state.runs.length){
          state.selectedRun = state.runs[0].run_id;
          if (runSelect){
            setSelectedOption(runSelect, state.selectedRun);
          }
        }
        if (state.selectedRun){
          fetchRunComparison(state.selectedRun).then(function(){ renderRunView(); });
        }
      } else {
        ensureCohortDefaults();
        renderCohortView();
      }
    }

    if (viewSelect){
      viewSelect.addEventListener('change', function(){
        state.selectedView = this.value;
        updateView();
      });
    }
    if (runSelect){
      runSelect.addEventListener('change', function(){
        state.selectedRun = this.value;
        updateView();
      });
    }
    if (runSelectAllBtn){
      runSelectAllBtn.addEventListener('click', function(){
        if (!runChecklist) return;
        var boxes = runChecklist.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(function(cb){ cb.checked = true; });
        state.selectedRuns = getSelectedRuns();
        renderCohortView();
      });
    }
    if (runClearBtn){
      runClearBtn.addEventListener('click', function(){
        if (!runChecklist) return;
        var boxes = runChecklist.querySelectorAll('input[type="checkbox"]');
        boxes.forEach(function(cb){ cb.checked = false; });
        state.selectedRuns = [];
        renderCohortView();
      });
    }
    if (groupingModeSelect){
      groupingModeSelect.addEventListener('change', function(){
        state.selectedGroupingMode = this.value;
        updateGroupingMode();
        if (state.selectedView === 'run'){
          renderRunView();
        } else {
          renderCohortView();
        }
      });
    }
    if (yearSelect){
      yearSelect.addEventListener('change', function(){
        state.selectedYear = parseInt(this.value, 10);
        if (state.selectedGroupingMode === 'year'){
          populateGradeSelect(state.selectedYear);
        }
        if (state.selectedView === 'run'){
          renderRunView();
        } else {
          renderCohortView();
        }
      });
    }
    if (gradeSelect){
      gradeSelect.addEventListener('change', function(){
        state.selectedGrade = this.value;
        if (state.selectedGroupingMode === 'grade'){
          populateYearsForGrade(state.selectedGrade);
        }
        if (state.selectedView === 'run'){
          renderRunView();
        } else {
          renderCohortView();
        }
      });
    }
    if (tableViewSelect){
      tableViewSelect.addEventListener('change', function(){
        state.selectedTableView = this.value;
        if (state.selectedView === 'run'){
          renderRunView();
        }
      });
    }
    
    var runSortSelect = $('#human-run-sort');
    if (runSortSelect){
      runSortSelect.addEventListener('change', function(){
        state.runTableSortOverride = null;
        updateRunTableSortIndicators();
        if (state.selectedView === 'run' && state.runData){
          renderRunTable(state.runData);
        }
      });
    }

    if (runTableHeaders && runTableHeaders.length){
      runTableHeaders.forEach(function(th){
        if (!th || !th.dataset.sortKey) return;
        if (!th.dataset.sortBound){
          th.dataset.sortBound = 'true';
          th.addEventListener('click', function(){
            if (state.selectedView !== 'run' || !state.runData) return;
            var sortKey = th.dataset.sortKey;
            var nextDirection = 'asc';
            if (state.runTableSortOverride && state.runTableSortOverride.column === sortKey){
              nextDirection = state.runTableSortOverride.direction === 'asc' ? 'desc' : 'asc';
            }
            state.runTableSortOverride = { column: sortKey, direction: nextDirection };
            updateRunTableSortIndicators();
            renderRunTable(state.runData);
          });
          th.addEventListener('keydown', function(event){
            if (event.key === 'Enter' || event.key === ' '){
              event.preventDefault();
              th.click();
            }
          });
        }
      });
    }
    
    var runFilterSelect = $('#human-run-filter');
    if (runFilterSelect){
      runFilterSelect.addEventListener('change', function(){
        if (state.selectedView === 'run' && state.runData){
          renderRunTable(state.runData);
        }
      });
    }
    if (cohortTypeSelect){
      cohortTypeSelect.addEventListener('change', function(){
        state.selectedCohortType = this.value;
        renderCohortView();
      });
    }
    
    var aggregationStrategySelect = $('#aggregation-strategy');
    if (aggregationStrategySelect){
      aggregationStrategySelect.addEventListener('change', function(){
        state.aggregationStrategy = this.value;
        state.runComparisons = {};
        state.runSummaries = {};
        state.cohortCache = {};
        state.cohortSummaries = {};
        state.currentRunSummary = null;
        state.currentRunSummaryKey = null;
        updateView();
      });
    }

    if (comparisonSourceToggle) {
      comparisonSourceToggle.addEventListener('click', function(event) {
        var target = event.target.closest('button');
        if (!target) return;
        var value = target.dataset.value;
        if (value === state.selectedComparisonSource) return;

        state.selectedComparisonSource = value;
        $$('button', comparisonSourceToggle).forEach(function(btn) {
          btn.classList.toggle('is-active', btn.dataset.value === value);
        });

        state.runSummaries = {};
        state.cohortSummaries = {};
        state.currentRunSummary = null;
        state.currentRunSummaryKey = null;
        updateBaselineHighlights();

        if (state.selectedView === 'run') {
          renderRunView();
        } else {
          // In the future, we might want to update the cohort view as well
        }
      });
    }

    populateRunSelectors();
    updateGroupingMode();
    updateView();
    updateBaselineHighlights();
  }

  document.addEventListener('DOMContentLoaded', function(){
    initThemeToggle();
    initOverview();
    initRunDetail();
    initCompare();
    initHumans();
    initAnalysis();
  });
})();
