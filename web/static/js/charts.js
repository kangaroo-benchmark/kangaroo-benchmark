(function(){
  if (!window.echarts){
    console.warn('ECharts library missing; charts initialization skipped.');
    window.DashboardCharts = function(){};
    window.DashboardCharts.prototype = {
      updateBreakdown: function(){},
      updateDistribution: function(){},
      updateHistogram: function(){},
      updatePredicted: function(){},
      updateConfusion: function(){},
      updateCompareSeries: function(){},
      clear: function(){}
    };
    return;
  }

  var BASE_PALETTE = ['#3b82f6', '#6366f1', '#f97316', '#22c55e', '#ef4444', '#06b6d4', '#a855f7'];

  function palette(count){
    var colors = [];
    for (var i = 0; i < count; i++){
      colors.push(BASE_PALETTE[i % BASE_PALETTE.length]);
    }
    return colors;
  }

  function readVar(name, fallback){
    var styles = window.getComputedStyle(document.documentElement);
    var value = styles.getPropertyValue(name);
    return value ? value.trim() : fallback;
  }

  function currentTheme(){
    return {
      text: readVar('--text', '#1f2126'),
      muted: readVar('--muted', '#5b6170'),
      border: readVar('--border', '#d7d9e0'),
      bg: readVar('--bg', '#f7f7f9'),
      elevated: readVar('--bg-elevated', '#ffffff'),
    };
  }

  function percentValue(value){
    if (value === null || value === undefined || isNaN(value)){
      return 0;
    }
    return Math.round(Number(value) * 10) / 10;
  }

  function percentLabel(value){
    if (value === null || value === undefined || isNaN(value)){
      return '0%';
    }
    return percentValue(value).toFixed(1) + '%';
  }

  function axisPercent(value){
    if (typeof value !== 'number' || !isFinite(value)){
      return '';
    }
    return Math.round(value).toString() + '%';
  }

  function rotateForLabels(labels){
    if (!labels || !labels.length) return 0;
    var maxLen = labels.reduce(function(max, label){
      return Math.max(max, (label || '').length);
    }, 0);
    if (maxLen > 18) return 60;
    if (maxLen > 12) return 40;
    if (maxLen > 8) return 25;
    return 0;
  }

  function wrapLabel(label, maxChars){
    if (!label) return '';
    var limit = maxChars || 12;
    if (label.length <= limit) return label;
    var chunks = [];
    var text = String(label);
    while (text.length > 0){
      chunks.push(text.slice(0, limit));
      text = text.slice(limit);
    }
    return chunks.join('\n');
  }

  function DashboardCharts(){
    this.instances = {};
    var self = this;
    this._resizeObserver = window.ResizeObserver ? new ResizeObserver(function(entries){
      entries.forEach(function(entry){
        var inst = self._findInstanceByElement(entry.target);
        if (inst){
          inst.chart.resize();
        }
      });
    }) : null;
  }

  DashboardCharts.prototype._findInstanceByElement = function(element){
    for (var key in this.instances){
      if (Object.prototype.hasOwnProperty.call(this.instances, key)){
        var inst = this.instances[key];
        if (inst && inst.element === element){
          return inst;
        }
      }
    }
    return null;
  };

  DashboardCharts.prototype._dispose = function(key){
    var inst = this.instances[key];
    if (!inst) return;
    if (this._resizeObserver){
      this._resizeObserver.unobserve(inst.element);
    }
    inst.chart.dispose();
    delete this.instances[key];
  };

  DashboardCharts.prototype._chartFor = function(key, element){
    if (!element) return null;
    var inst = this.instances[key];
    if (inst && inst.element !== element){
      this._dispose(key);
      inst = null;
    }
    if (!inst){
      var chart = window.echarts.init(element, null, {
        renderer: 'canvas',
        useDirtyRect: true,
      });
      inst = { chart: chart, element: element };
      this.instances[key] = inst;
      if (this._resizeObserver){
        this._resizeObserver.observe(element);
      }
    }
    return inst.chart;
  };

  DashboardCharts.prototype._applyOption = function(key, element, option){
    var chart = this._chartFor(key, element);
    if (!chart) return;
    chart.clear();
    chart.setOption(option, { notMerge: true, lazyUpdate: true });
    if (typeof requestAnimationFrame === 'function'){
      requestAnimationFrame(function(){ chart.resize(); });
    } else {
      setTimeout(function(){ chart.resize(); }, 0);
    }
  };

  DashboardCharts.prototype._emptyOption = function(message){
    var theme = currentTheme();
    return {
      animation: false,
      title: {
        text: message || 'No data available',
        left: 'center',
        top: 'middle',
        textStyle: {
          color: theme.muted,
          fontSize: 14,
          fontWeight: 'normal',
        },
      },
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      xAxis: { show: false },
      yAxis: { show: false },
      series: [],
    };
  };

  DashboardCharts.prototype.updateBreakdown = function(element, key, breakdown){
    if (!element) return;
    var labels = Object.keys(breakdown || {});
    if (!labels.length){
      this._applyOption(key, element, this._emptyOption('No breakdown data'));
      return;
    }
    var values = labels.map(function(label){
      var entry = breakdown[label] || {};
      return percentValue((entry.accuracy || 0) * 100);
    });
    var theme = currentTheme();
    var option = {
      color: palette(values.length),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        valueFormatter: function(value){ return percentLabel(value); },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 16, containLabel: true },
      xAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: {
          color: theme.muted,
          formatter: axisPercent,
        },
        splitLine: {
          lineStyle: {
            color: theme.border,
            opacity: 0.35,
          }
        }
      },
      yAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          color: theme.text,
          interval: 0,
          hideOverlap: true,
          formatter: function(value){ return wrapLabel(value, 14); },
        },
        axisTick: { alignWithLabel: true }
      },
      series: [{
        type: 'bar',
        data: values,
        barMaxWidth: 28,
        label: {
          show: true,
          position: 'right',
          formatter: function(params){ return percentLabel(params.value); },
          color: theme.text,
          fontWeight: 600,
        },
        itemStyle: {
          borderRadius: [4, 4, 4, 4],
        },
      }],
    };
    this._applyOption(key, element, option);
  };

  DashboardCharts.prototype.updateDistribution = function(element, key, entries){
    if (!element) return;
    var list = entries || [];
    var chartKey = key || (element && element.id) || 'distribution';
    if (!list.length){
      this._applyOption(chartKey, element, this._emptyOption('No distribution data'));
      return;
    }
    var labels = list.map(function(entry){ return entry && entry.label ? entry.label : 'Unknown'; });
    var counts = list.map(function(entry){ return entry && entry.count ? Number(entry.count) : 0; });
    var theme = currentTheme();
    var rotation = rotateForLabels(labels);
    var option = {
      color: palette(Math.min(counts.length, 6)),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: function(params){
          if (!params || !params.length) return '';
          var idx = params[0].dataIndex;
          var current = list[idx] || {};
          var percent = current.percentage ? (Number(current.percentage) * 100) : 0;
          var percentLabel = percent ? percent.toFixed(1).replace(/\.0$/, '') + '%' : '0%';
          return params[0].name + ': ' + params[0].value + ' (' + percentLabel + ')';
        }
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: rotation ? 70 : 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          color: theme.text,
          interval: 0,
          rotate: rotation,
          hideOverlap: true,
          formatter: function(value){ return wrapLabel(value, 12); }
        },
        axisTick: { alignWithLabel: true }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: theme.muted },
        splitLine: {
          lineStyle: { color: theme.border, opacity: 0.35 }
        }
      },
      series: [{
        type: 'bar',
        data: counts,
        barMaxWidth: 40,
        itemStyle: { borderRadius: [4, 4, 0, 0] },
        label: {
          show: true,
          position: 'top',
          color: theme.muted,
          formatter: function(params){ return params.value; }
        }
      }]
    };
    this._applyOption(chartKey, element, option);
  };

  DashboardCharts.prototype.updateHistogram = function(element, histogram){
    if (!element) return;
    if (!histogram || !histogram.counts || !histogram.counts.length){
      this._applyOption(element.id, element, this._emptyOption('No distribution data'));
      return;
    }
    var labels = [];
    for (var i = 0; i < histogram.counts.length; i++){
      var start = Math.round(histogram.bins[i] || 0);
      var end = Math.round(histogram.bins[i + 1] || start);
      labels.push(start + '–' + end);
    }
    var counts = histogram.counts.map(function(v){ return Number(v) || 0; });
    var theme = currentTheme();
    var key = element.id;
    var option = {
      color: [palette(1)[0]],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: labels.length > 8 ? 60 : 30, containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          color: theme.text,
          interval: 0,
          rotate: rotateForLabels(labels),
          hideOverlap: true,
          formatter: function(value){ return wrapLabel(value, 10); },
        },
        axisTick: { alignWithLabel: true },
        nameGap: 16,
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: theme.muted },
        splitLine: {
          lineStyle: { color: theme.border, opacity: 0.35 },
        },
      },
      series: [{
        type: 'bar',
        data: counts,
        barMaxWidth: 32,
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
        label: {
          show: true,
          position: 'top',
          color: theme.muted,
          formatter: '{c}',
        },
      }],
    };
    this._applyOption(key, element, option);
  };

  DashboardCharts.prototype.updatePredicted = function(element, counts){
    if (!element) return;
    var labels = Object.keys(counts || {});
    if (!labels.length){
      this._applyOption(element.id, element, this._emptyOption('No predictions recorded'));
      return;
    }
    var data = labels.map(function(label){
      return { name: label, value: counts[label] || 0 };
    });
    var theme = currentTheme();
    var option = {
      color: palette(labels.length),
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
      },
      legend: {
        bottom: 0,
        type: 'scroll',
        textStyle: { color: theme.muted },
      },
      series: [{
        type: 'pie',
        radius: ['45%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 6,
          borderColor: theme.elevated,
          borderWidth: 2,
        },
        label: {
          color: theme.text,
        },
        labelLine: {
          length: 18,
          length2: 12,
        },
        data: data,
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateConfusion = function(element, matrix){
    if (!element) return;
    var labels = Object.keys(matrix || {}).sort();
    if (!labels.length){
      this._applyOption(element.id, element, this._emptyOption('No confusion data'));
      return;
    }
    var data = [];
    var maxVal = 0;
    for (var r = 0; r < labels.length; r++){
      var rowKey = labels[r];
      var row = matrix[rowKey] || {};
      for (var c = 0; c < labels.length; c++){
        var colKey = labels[c];
        var value = Number(row[colKey] || 0);
        data.push([c, r, value]);
        if (value > maxVal){
          maxVal = value;
        }
      }
    }
    var theme = currentTheme();
    var gradient = [readVar('--bg-hover', '#eef2ff'), palette(1)[0]];
    var option = {
      tooltip: {
        position: 'top',
        formatter: function(params){
          var actual = labels[params.value[1]];
          var predicted = labels[params.value[0]];
          return 'Actual ' + actual + ' / Predicted ' + predicted + ': ' + params.value[2];
        },
      },
      grid: { left: '12%', right: '6%', top: 40, bottom: 60 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: theme.text },
        splitArea: { show: true },
        splitLine: { show: true, lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      yAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: theme.text },
        splitArea: { show: true },
        splitLine: { show: true, lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      visualMap: {
        min: 0,
        max: maxVal || 1,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        inRange: {
          color: gradient,
        },
        textStyle: { color: theme.muted },
      },
      series: [{
        name: 'Confusion',
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          color: theme.text,
          formatter: function(params){
            return params.value[2] ? String(params.value[2]) : '';
          },
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.35)',
          },
        },
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateCompareSeries = function(element, seriesData){
    if (!element) return;
    var entries = (seriesData || []).filter(function(entry){ return entry && entry.metric; });
    if (!entries.length){
      this._applyOption(element.id, element, this._emptyOption('No comparison data'));
      return;
    }
    var categories = entries.map(function(entry){ return entry.metric; });
    var values = entries.map(function(entry){
      if (entry.delta === null || entry.delta === undefined || isNaN(entry.delta)){
        return 0;
      }
      return Math.round(Number(entry.delta) * 1000) / 1000;
    });
    var theme = currentTheme();
    var data = values.map(function(value){
      return {
        value: value,
        itemStyle: { color: value >= 0 ? palette(1)[0] : '#ef4444' },
        label: {
          position: value >= 0 ? 'top' : 'bottom'
        }
      };
    });
    var option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        valueFormatter: function(value){
          if (value === null || value === undefined) return '0';
          return Number(value).toFixed(3);
        },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: {
          color: theme.text,
          interval: 0,
          rotate: rotateForLabels(categories),
          hideOverlap: true,
          formatter: function(value){
            var normalised = (value || '').replace(/_/g, ' ');
            return wrapLabel(normalised, 12);
          },
        },
        axisTick: { alignWithLabel: true },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: theme.muted },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      series: [{
        type: 'bar',
        data: data,
        barMaxWidth: 32,
        label: {
          show: true,
          color: theme.text,
          formatter: function(params){
            return Number(params.value).toFixed(3);
          },
        },
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          data: [{ yAxis: 0 }],
          lineStyle: {
            color: theme.border,
            type: 'dashed',
          }
        }
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateScatter = function(element, data) {
    if (!element) return;
    if (!data || !data.length) {
      this._applyOption(element.id, element, this._emptyOption('No data for scatter plot'));
      return;
    }
    var theme = currentTheme();
    var option = {
      color: palette(1),
      tooltip: {
        trigger: 'item',
        formatter: function(params) {
          var data = params.data || [];
          return 'ID: ' + data[2] + '<br/>' +
                 'Human: ' + percentLabel(data[0] * 100) + '<br/>' +
                 'LLM: ' + percentLabel(data[1] * 100) + '<br/>' +
                 'Disagreement: ' + (data[3] ? data[3].toFixed(3) : 'N/A');
        }
      },
      grid: { left: '10%', right: '10%', top: 60, bottom: 60 },
      xAxis: {
        type: 'value',
        name: 'Human Correctness (%)',
        nameLocation: 'middle',
        nameGap: 30,
        min: 0,
        max: 1,
        axisLabel: {
          color: theme.muted,
          formatter: function(v) { return Math.round(v * 100) + '%'; }
        },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.35 } }
      },
      yAxis: {
        type: 'value',
        name: 'Average LLM Correctness (%)',
        nameLocation: 'middle',
        nameGap: 50,
        min: 0,
        max: 1,
        axisLabel: {
          color: theme.muted,
          formatter: function(v) { return Math.round(v * 100) + '%'; }
        },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.35 } }
      },
      series: [{
        type: 'scatter',
        symbolSize: 8,
        data: data
      }]
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateBar = function(element, data) {
    if (!element) return;
    if (!data || !data.length) {
      this._applyOption(element.id, element, this._emptyOption('No data for bar chart'));
      return;
    }
    var theme = currentTheme();
    var years = data.map(function(item) { return item.year; });
    var humanScores = data.map(function(item) { return item.avg_human_score; });
    var llmScores = data.map(function(item) { return item.avg_llm_score; });
    var normalizedScores = data.map(function(item) { return item.normalized_human_score; });

    var option = {
      color: palette(3),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      legend: {
        data: ['Human Score', 'LLM Score', 'Normalized Human Score'],
        textStyle: { color: theme.muted }
      },
      grid: { left: '10%', right: '10%', top: 60, bottom: 60 },
      xAxis: {
        type: 'category',
        data: years,
        axisLabel: { color: theme.text }
      },
      yAxis: [
        {
          type: 'value',
          name: 'Score',
          min: 0,
          max: 1,
          axisLabel: { color: theme.muted, formatter: function(v) { return Math.round(v * 100) + '%'; } }
        },
        {
          type: 'value',
          name: 'Normalized',
          min: 0,
          axisLabel: { color: theme.muted }
        }
      ],
      series: [
        {
          name: 'Human Score',
          type: 'bar',
          data: humanScores
        },
        {
          name: 'LLM Score',
          type: 'bar',
          data: llmScores
        },
        {
          name: 'Normalized Human Score',
          type: 'line',
          yAxisIndex: 1,
      data: normalizedScores
    }
  ]
};
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateCDF = function(element, humanSeries, markers, maxScore){
    if (!element) return;
    var series = Array.isArray(humanSeries) ? humanSeries.slice() : [];
    if (!series.length){
      this._applyOption(element.id, element, this._emptyOption('No CDF data'));
      return;
    }
    var theme = currentTheme();
    var lineData = series.map(function(point){
      return [Number(point.score || 0), Number(point.percentile || 0) * 100];
    });
    var maxX = typeof maxScore === 'number' && isFinite(maxScore) ? maxScore : lineData.reduce(function(acc, value){ return Math.max(acc, value[0]); }, 0);
    var scatterData = (markers || []).map(function(marker){
      return {
        value: [Number(marker.score || 0), Number(marker.percentile || 0) * 100],
        name: marker.label || 'Run',
      };
    });
    var option = {
      color: palette(4),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line' },
        formatter: function(params){
          if (!params || !params.length) return '';
          var entries = params.map(function(item){
            return item.seriesName + ': ' + item.value[0].toFixed(1) + ' · ' + item.value[1].toFixed(1) + '%';
          });
          return entries.join('<br/>');
        }
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 50, containLabel: true },
      xAxis: {
        type: 'value',
        min: 0,
        max: Math.max(maxX, 1),
        axisLabel: { color: theme.muted },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.25 } }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: theme.muted, formatter: function(value){ return value + '%'; } },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.3 } }
      },
      series: [
        {
          name: 'Human CDF',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: lineData,
          lineStyle: { width: 2 }
        },
        {
          name: 'LLM',
          type: 'scatter',
          symbolSize: 12,
          label: {
            show: true,
            color: theme.text,
            formatter: function(params){ return params.name; },
            position: 'top'
          },
          data: scatterData
        }
      ]
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateHeatmap = function(element, payload){
    if (!element) return;
    if (!payload || !payload.xLabels || !payload.xLabels.length || !payload.yLabels || !payload.yLabels.length){
      this._applyOption(element.id, element, this._emptyOption('No heatmap data'));
      return;
    }
    var data = Array.isArray(payload.values) ? payload.values : [];
    var theme = currentTheme();
    var maxAbs = data.reduce(function(acc, item){
      var value = item && item.length > 2 ? Math.abs(item[2]) : 0;
      return Math.max(acc, value);
    }, 0);
    if (!isFinite(maxAbs) || maxAbs === 0){
      maxAbs = 1;
    }
    var option = {
      tooltip: {
        position: 'top',
        formatter: function(params){
          var grade = payload.yLabels[params.value[1]];
          var bin = payload.xLabels[params.value[0]];
          return grade + ' · ' + bin + ': ' + params.value[2].toFixed(2) + ' pts';
        }
      },
      grid: { left: '12%', right: '6%', top: 40, bottom: 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: payload.xLabels,
        axisLabel: { color: theme.text, hideOverlap: true, rotate: rotateForLabels(payload.xLabels) },
        splitArea: { show: false },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'category',
        data: payload.yLabels,
        axisLabel: { color: theme.text },
        splitArea: { show: false },
        splitLine: { show: false }
      },
      visualMap: {
        min: -maxAbs,
        max: maxAbs,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        inRange: { color: ['#ef4444', '#ffffff', palette(1)[0]] },
        textStyle: { color: theme.muted }
      },
      series: [
        {
          name: 'Delta',
          type: 'heatmap',
          data: data,
          label: { show: false }
        }
      ]
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateBoxViolin = function(element, data){
    if (!element) return;
    if (!data || !data.length){
      this._applyOption(element.id, element, this._emptyOption('No percentile data'));
      return;
    }
    var theme = currentTheme();
    var categories = data.map(function(item){ return item.label || 'Grade'; });
    var seriesData = data.map(function(item){
      var stats = item.stats || {};
      var median = Number(stats.median || 0);
      var min = Number((stats.min !== undefined) ? stats.min : median);
      var p25 = Number((stats.p25 !== undefined) ? stats.p25 : median);
      var p75 = Number((stats.p75 !== undefined) ? stats.p75 : median);
      var max = Number((stats.max !== undefined) ? stats.max : median);
      return [min * 100, p25 * 100, median * 100, p75 * 100, max * 100];
    });
    var option = {
      color: palette(1),
      tooltip: {
        trigger: 'item',
        formatter: function(params){
          if (!params || !params.value) return '';
          var values = params.value.map(function(v){ return Number(v).toFixed(1) + '%'; });
          return params.name + '<br/>[' + values.join(' · ') + ']';
        }
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 50, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { color: theme.text }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: theme.muted, formatter: function(value){ return value + '%'; } },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.3 } }
      },
      series: [
        {
          name: 'Percentiles',
          type: 'boxplot',
          data: seriesData,
          itemStyle: { color: palette(1)[0] }
        }
      ]
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.clear = function(){
    for (var key in this.instances){
      if (Object.prototype.hasOwnProperty.call(this.instances, key)){
        this._dispose(key);
      }
    }
    this.instances = {};
  };

  window.DashboardCharts = DashboardCharts;
})();
