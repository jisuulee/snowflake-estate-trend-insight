const charts = {};
let modelsData = [];

function buildOptgroups(grouped) {
  return Object.entries(grouped)
    .map(
      ([prefix, names]) =>
        `<optgroup label="${prefix}">${names.map((n) => `<option value="${n}">${n}</option>`).join('')}</optgroup>`,
    )
    .join('');
}

async function loadTables() {
  try {
    const data = await apiFetch('/api/tables');
    const featureOpts = buildOptgroups(data.feature_grouped);
    document.getElementById('predictTable').innerHTML = featureOpts;
    document.getElementById('insightsTable').innerHTML = featureOpts;
    document.getElementById('exploreTable').innerHTML = buildOptgroups(data.grouped);
    loadFilters('predictTable', 'predictState', 'predictCity');
    loadFilters('exploreTable', 'exploreState', null);
  } catch (e) {
    console.error('테이블 목록 로드 실패:', e);
  }
}

async function loadModels() {
  try {
    const data = await apiFetch('/api/models');
    modelsData = data.models;
    const opts = modelsData
      .map((m) => `<option value="${m.name}">${m.name}</option>`)
      .join('');
    document.getElementById('predictModelName').innerHTML = opts;
    document.getElementById('insightsModelName').innerHTML = opts;
    onModelNameChange();
    onInsightsModelNameChange();
  } catch (e) {
    console.error('모델 목록 로드 실패:', e);
  }
}

function onModelNameChange() {
  const name = document.getElementById('predictModelName').value;
  const model = modelsData.find((m) => m.name === name);
  const verSel = document.getElementById('predictModelVersion');
  verSel.innerHTML = model
    ? model.versions.map((v) => `<option value="${v}">${v}</option>`).join('')
    : '<option value="">-</option>';
}

function onInsightsModelNameChange() {
  const name = document.getElementById('insightsModelName').value;
  const model = modelsData.find((m) => m.name === name);
  const verSel = document.getElementById('insightsModelVersion');
  verSel.innerHTML = model
    ? model.versions.map((v) => `<option value="${v}">${v}</option>`).join('')
    : '<option value="">-</option>';
}

async function runInsights() {
  const btn = document.getElementById('btnInsights');
  const loading = document.getElementById('insightsLoading');
  const errorEl = document.getElementById('insightsError');
  const content = document.getElementById('insightsContent');

  btn.disabled = true;
  loading.style.display = 'flex';
  errorEl.style.display = 'none';
  content.style.display = 'none';

  const body = {
    table: document.getElementById('insightsTable').value,
    model_name: document.getElementById('insightsModelName').value,
    model_version: document.getElementById('insightsModelVersion').value,
    limit: 5000,
  };

  try {
    const data = await apiFetch('/api/insights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (data.status === 'empty') {
      errorEl.textContent = '데이터가 없습니다.';
      errorEl.style.display = 'block';
      return;
    }

    if (data.fgi_by_state) renderFgiStateChart(data.fgi_by_state);
    if (data.price_top10_up) renderHbarChart('priceUpChart', data.price_top10_up, 'rgba(78,205,196,0.75)', '집값 변화율 (%)');
    if (data.price_top10_down) renderHbarChart('priceDownChart', data.price_top10_down, 'rgba(255,107,107,0.75)', '집값 변화율 (%)');
    if (data.contract_top10_up) renderHbarChart('contractUpChart', data.contract_top10_up, 'rgba(78,205,196,0.75)', '거래량 변화율 (%)');
    if (data.contract_top10_down) renderHbarChart('contractDownChart', data.contract_top10_down, 'rgba(255,107,107,0.75)', '거래량 변화율 (%)');

    content.style.display = 'block';
  } catch (e) {
    errorEl.textContent = '분석 실패: ' + e.message;
    errorEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}

function renderFgiStateChart(fgi) {
  destroyChart('fgiState');
  const ctx = document.getElementById('fgiStateChart').getContext('2d');
  charts['fgiState'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: fgi.labels,
      datasets: [{
        label: 'FGI (평균)',
        data: fgi.values,
        backgroundColor: fgi.values.map((v) =>
          v >= 60 ? 'rgba(103, 137, 230, 1)' : v >= 40 ? 'rgba(255,193,7,0.75)' : 'rgba(78,205,196,0.75)',
        ),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const v = ctx.raw;
              return v >= 60 ? '공포 구간' : v >= 40 ? '중립' : '탐욕 구간';
            },
          },
        },
      },
      scales: {
        y: { min: 500, max: 2000, title: { display: true, text: 'FGI' } },
      },
    },
  });
}

function renderHbarChart(canvasId, chartData, color, xLabel) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: chartData.labels,
      datasets: [{
        data: chartData.values,
        backgroundColor: color,
        borderWidth: 1,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: xLabel } },
        y: { ticks: { font: { size: 11 } } },
      },
    },
  });
}

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || '요청 실패');
  }
  return resp.json();
}

async function checkHealth() {
  const el = document.getElementById('connectionStatus');
  try {
    const data = await apiFetch('/api/health');
    if (data.snowflake === 'connected') {
      el.textContent = '연결됨';
      el.className = 'nav-status connected';
    } else {
      el.textContent = '연결 실패';
      el.className = 'nav-status disconnected';
    }
  } catch {
    el.textContent = '서버 오류';
    el.className = 'nav-status disconnected';
  }
}

let filtersCache = {};

async function loadFilters(tableSelectId, stateSelectId, citySelectId) {
  const table = document.getElementById(tableSelectId).value;
  const stateSelect = document.getElementById(stateSelectId);
  const citySelect = citySelectId ? document.getElementById(citySelectId) : null;

  try {
    const data = await apiFetch('/api/filters/' + table);
    filtersCache = data;

    stateSelect.innerHTML = '<option value="">전체</option>';
    data.states.forEach((s) => {
      stateSelect.innerHTML += `<option value="${s}">${s}</option>`;
    });

    if (citySelect) {
      citySelect.innerHTML = '<option value="">전체</option>';
    }
  } catch (e) {
    console.error('필터 로드 실패:', e);
  }
}

document.getElementById('predictTable').addEventListener('change', () => {
  loadFilters('predictTable', 'predictState', 'predictCity');
});

document.getElementById('predictState').addEventListener('change', () => {
  const state = document.getElementById('predictState').value;
  const citySelect = document.getElementById('predictCity');
  citySelect.innerHTML = '<option value="">전체</option>';
  if (state && filtersCache.cities_by_state && filtersCache.cities_by_state[state]) {
    filtersCache.cities_by_state[state].forEach((c) => {
      citySelect.innerHTML += `<option value="${c}">${c}</option>`;
    });
  }
});

document.getElementById('exploreTable').addEventListener('change', () => {
  loadFilters('exploreTable', 'exploreState', null);
});

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

async function runPredict() {
  const btn = document.getElementById('btnPredict');
  const loading = document.getElementById('predictLoading');
  const errorEl = document.getElementById('predictError');

  btn.disabled = true;
  loading.style.display = 'flex';
  errorEl.style.display = 'none';
  document.getElementById('metricsRow').style.display = 'none';
  document.getElementById('metricsDesc').style.display = 'none';
  document.getElementById('chartsRow').style.display = 'none';
  document.getElementById('chartsRow2').style.display = 'none';
  document.getElementById('predictTable-container').style.display = 'none';

  const body = {
    table: document.getElementById('predictTable').value,
    state: document.getElementById('predictState').value || null,
    city: document.getElementById('predictCity').value || null,
    limit: parseInt(document.getElementById('predictLimit').value) || 500,
    model_name: document.getElementById('predictModelName').value,
    model_version: document.getElementById('predictModelVersion').value,
  };

  try {
    const data = await apiFetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (data.status === 'empty') {
      errorEl.textContent = data.message;
      errorEl.style.display = 'block';
      return;
    }

    if (data.metrics) {
      document.getElementById('metricMAE').textContent = data.metrics.mae;
      document.getElementById('metricRMSE').textContent = data.metrics.rmse;
      document.getElementById('metricR2').textContent = data.metrics.r2;
      document.getElementById('metricCount').textContent = data.metrics.count + '건';
      document.getElementById('metricsRow').style.display = 'flex';
      document.getElementById('metricsDesc').style.display = 'block';
    }

    if (data.chart_data) {
      document.getElementById('chartsRow').style.display = 'grid';
      document.getElementById('chartsRow2').style.display = 'grid';
      renderScatterChart(data.chart_data);
      renderResidualChart(data.chart_data);
      if (data.chart_data.monthly) renderMonthlyChart(data.chart_data.monthly);
      if (data.chart_data.by_state) renderStateChart(data.chart_data.by_state);
    }

    if (data.data && data.data.length > 0) {
      renderTable(data.data, data.total_rows);
      document.getElementById('predictTable-container').style.display = 'block';
    }
  } catch (e) {
    errorEl.textContent = '예측 실패: ' + e.message;
    errorEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}

function renderScatterChart(chartData) {
  destroyChart('scatter');
  const ctx = document.getElementById('scatterChart').getContext('2d');
  const points = chartData.actual.map((a, i) => ({ x: a, y: chartData.predicted[i] }));
  const maxVal = Math.max(...chartData.actual, ...chartData.predicted);

  charts['scatter'] = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Predicted vs Actual',
          data: points,
          backgroundColor: 'rgba(78,205,196,0.5)',
          pointRadius: 3,
        },
        {
          label: 'Perfect',
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          type: 'line',
          borderColor: '#FF6B6B',
          borderDash: [5, 5],
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { title: { display: true, text: 'Actual' } },
        y: { title: { display: true, text: 'Predicted' } },
      },
    },
  });
}

function renderResidualChart(chartData) {
  destroyChart('residual');
  const ctx = document.getElementById('residualChart').getContext('2d');
  const residuals = chartData.actual.map((a, i) => a - chartData.predicted[i]);

  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const binCount = 30;
  const binSize = (max - min) / binCount || 1;
  const bins = Array(binCount).fill(0);
  const labels = [];
  for (let i = 0; i < binCount; i++) {
    labels.push(Math.round(min + i * binSize));
    residuals.forEach((r) => {
      const idx = Math.min(Math.floor((r - min) / binSize), binCount - 1);
      if (idx === i) bins[i]++;
    });
  }

  charts['residual'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '잔차 빈도',
        data: bins,
        backgroundColor: 'rgba(255,107,107,0.6)',
        borderColor: 'rgba(255,107,107,1)',
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Residual (Actual - Predicted)' } },
        y: { title: { display: true, text: 'Count' } },
      },
    },
  });
}

function renderMonthlyChart(monthly) {
  destroyChart('monthly');
  const ctx = document.getElementById('monthlyChart').getContext('2d');
  charts['monthly'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: monthly.labels,
      datasets: [
        {
          label: '실제 평균',
          data: monthly.actual_avg,
          borderColor: '#FF6B6B',
          backgroundColor: 'rgba(255,107,107,0.1)',
          fill: true,
          tension: 0.3,
        },
        {
          label: '예측 평균',
          data: monthly.predicted_avg,
          borderColor: '#4ECDC4',
          backgroundColor: 'rgba(78,205,196,0.1)',
          fill: true,
          tension: 0.3,
          borderDash: [5, 5],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { ticks: { maxRotation: 45 } },
        y: { title: { display: true, text: '평균 거래량' } },
      },
    },
  });
}

function renderStateChart(byState) {
  destroyChart('state');
  const ctx = document.getElementById('stateChart').getContext('2d');
  charts['state'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: byState.labels,
      datasets: [
        {
          label: '실제 평균',
          data: byState.actual_avg,
          backgroundColor: 'rgba(255,107,107,0.7)',
        },
        {
          label: '예측 평균',
          data: byState.predicted_avg,
          backgroundColor: 'rgba(78,205,196,0.7)',
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { ticks: { maxRotation: 45 } },
        y: { title: { display: true, text: '평균 거래량' } },
      },
    },
  });
}

function renderTable(data, totalRows) {
  const head = document.getElementById('resultHead');
  const body = document.getElementById('resultBody');
  document.getElementById('tableRowCount').textContent = `(${data.length} / ${totalRows}건)`;

  const cols = Object.keys(data[0]);
  head.innerHTML = '<tr>' + cols.map((c) => `<th>${c}</th>`).join('') + '</tr>';
  body.innerHTML = data
    .map(
      (row) =>
        '<tr>' +
        cols.map((c) => {
          let val = row[c];
          if (typeof val === 'number') val = val.toFixed ? val.toFixed(2) : val;
          return `<td>${val ?? ''}</td>`;
        }).join('') +
        '</tr>',
    )
    .join('');
}

async function loadExploreData() {
  const content = document.getElementById('exploreContent');
  const loading = document.getElementById('exploreLoading');
  const errorEl = document.getElementById('exploreError');

  loading.style.display = 'flex';
  errorEl.style.display = 'none';
  content.innerHTML = '';

  const table = document.getElementById('exploreTable').value;
  const state = document.getElementById('exploreState').value;
  const limit = document.getElementById('exploreLimit').value;

  let url = `/api/data/${table}?limit=${limit}`;
  if (state) url += `&state=${encodeURIComponent(state)}`;

  try {
    const data = await apiFetch(url);

    let html = `<p>총 ${data.total_rows}건 중 ${data.returned_rows}건 표시</p>`;

    if (data.stats) {
      html += `<h3>기초 통계량</h3><div class="table-wrapper"><table><thead><tr><th>컬럼</th><th>평균</th><th>표준편차</th><th>최소</th><th>최대</th></tr></thead><tbody>`;
      for (const [col, s] of Object.entries(data.stats)) {
        html += `<tr><td>${col}</td><td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.max}</td></tr>`;
      }
      html += `</tbody></table></div>`;
    }

    if (data.data.length > 0) {
      const cols = data.columns.slice(0, 15);
      html += `<h3>데이터 미리보기</h3><div class="table-wrapper"><table><thead><tr>${cols.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
      data.data.forEach((row) => {
        html +=
          '<tr>' +
          cols.map((c) => {
            let v = row[c];
            if (typeof v === 'number') v = v.toFixed ? v.toFixed(4) : v;
            return `<td>${v ?? ''}</td>`;
          }).join('') +
          '</tr>';
      });
      html += `</tbody></table></div>`;
    }

    content.innerHTML = html;
  } catch (e) {
    errorEl.textContent = '데이터 조회 실패: ' + e.message;
    errorEl.style.display = 'block';
  } finally {
    loading.style.display = 'none';
  }
}

checkHealth();
loadModels();
loadTables();
