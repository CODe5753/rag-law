// 경과 시간 타이머
let _timer = null;

function startTimer(elId = 'elapsed') {
  const el = document.getElementById(elId);
  if (!el) return;
  let sec = 0;
  el.textContent = '0';
  stopTimer();
  _timer = setInterval(() => {
    sec++;
    el.textContent = sec;
  }, 1000);
}

function stopTimer() {
  if (_timer) { clearInterval(_timer); _timer = null; }
}

// HTMX 이벤트 (샘플 버튼 / 비교 버튼용)
document.addEventListener('htmx:beforeRequest', (e) => {
  if (e.detail.elt.dataset.qid || e.detail.elt.dataset.compareQid) {
    startTimer('elapsed');
    const txt = document.getElementById('loading-text');
    if (txt) txt.textContent = e.detail.elt.dataset.compareQid
      ? '두 파이프라인 캐시 비교 중…'
      : '캐시 로드 중…';
  }
});

document.addEventListener('htmx:afterRequest', stopTimer);

document.addEventListener('htmx:responseError', (e) => {
  stopTimer();
  const target = document.getElementById('result-area');
  if (target && e.detail.xhr?.responseText) {
    target.innerHTML = e.detail.xhr.responseText;
  }
});

document.addEventListener('htmx:sendError', () => {
  stopTimer();
  const target = document.getElementById('result-area');
  if (target) {
    target.innerHTML = "<div class='bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700'><strong>서버에 연결할 수 없습니다.</strong><p class='mt-1 text-xs text-red-500'>서버가 실행 중인지 확인하세요.</p></div>";
  }
});

// 샘플 버튼 동적 URL 설정
function getSampleUrl(qid) {
  const pipeline = document.querySelector('input[name="pipeline"]:checked')?.value || 'crag';
  return `/samples/${qid}?pipeline=${pipeline}`;
}

// ── SSE 스트리밍 (자유 입력 폼 전용) ──────────────────────────

const NODE_LABELS = {
  retrieve: "법령 문서 검색 중 (Dense+BM25 하이브리드)",
  grade_documents: "관련도 채점 중 (BGE-Reranker)",
  generate: "AI 답변 생성 중",
  fallback: "Fallback 응답 생성 중 (관련 조문 부족)",
};
const NODE_DONE_LABELS = {
  retrieve: "검색 완료",
  grade_documents: "채점 완료",
  generate: "답변 생성 완료",
  fallback: "Fallback 완료",
};

let _streamTimer = null;
let _currentES = null;

function resetStreamUI() {
  const box = document.getElementById('stream-indicator');
  const steps = document.getElementById('stream-steps');
  const elapsed = document.getElementById('stream-elapsed');
  if (steps) steps.innerHTML = '';
  if (elapsed) elapsed.textContent = '0';
  if (box) box.classList.add('hidden');
  if (_streamTimer) { clearInterval(_streamTimer); _streamTimer = null; }
}

function showStreamUI(pipeline) {
  const box = document.getElementById('stream-indicator');
  if (box) box.classList.remove('hidden');
  const label = document.getElementById('stream-label');
  if (label) label.textContent = pipeline === 'crag' ? 'LangGraph 실행 중' : 'Hybrid Qdrant 실행 중';
  const elapsed = document.getElementById('stream-elapsed');
  if (elapsed) elapsed.textContent = '0';
  let sec = 0;
  if (_streamTimer) clearInterval(_streamTimer);
  _streamTimer = setInterval(() => {
    sec++;
    if (elapsed) elapsed.textContent = sec;
  }, 1000);
}

function addStreamStep(node) {
  const steps = document.getElementById('stream-steps');
  if (!steps) return;
  // 이전 단계들 완료 처리
  steps.querySelectorAll('li[data-active="true"]').forEach(li => {
    li.dataset.active = 'false';
    const prevNode = li.dataset.node;
    li.innerHTML = `<span class="text-green-600">✓</span> <span class="text-gray-600">${NODE_DONE_LABELS[prevNode] || prevNode}</span>`;
  });
  const li = document.createElement('li');
  li.dataset.node = node;
  li.dataset.active = 'true';
  li.className = 'flex items-center gap-2';
  li.innerHTML = `<span class="text-blue-600 animate-pulse">●</span> <span class="text-blue-700 font-medium">${NODE_LABELS[node] || node}</span>`;
  steps.appendChild(li);
}

function finalizeStreamSteps() {
  const steps = document.getElementById('stream-steps');
  if (!steps) return;
  steps.querySelectorAll('li[data-active="true"]').forEach(li => {
    const node = li.dataset.node;
    li.dataset.active = 'false';
    li.innerHTML = `<span class="text-green-600">✓</span> <span class="text-gray-600">${NODE_DONE_LABELS[node] || node}</span>`;
  });
}

function startStreamQuery(question, pipeline) {
  if (_currentES) { _currentES.close(); _currentES = null; }
  resetStreamUI();
  showStreamUI(pipeline);

  const url = `/query/stream?question=${encodeURIComponent(question)}&pipeline=${encodeURIComponent(pipeline)}`;
  const es = new EventSource(url);
  _currentES = es;

  es.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (data.type === 'node') {
      addStreamStep(data.node);
    } else if (data.type === 'done') {
      finalizeStreamSteps();
      const target = document.getElementById('result-area');
      if (target) target.innerHTML = data.html;
      es.close();
      _currentES = null;
      if (_streamTimer) { clearInterval(_streamTimer); _streamTimer = null; }
      const box = document.getElementById('stream-indicator');
      if (box) box.classList.add('hidden');
    } else if (data.type === 'error') {
      const target = document.getElementById('result-area');
      if (target) {
        target.innerHTML = `<div class='bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700'><strong>파이프라인 오류</strong><p class='mt-1 text-xs'>${data.message}</p></div>`;
      }
      es.close();
      _currentES = null;
      if (_streamTimer) { clearInterval(_streamTimer); _streamTimer = null; }
      const box = document.getElementById('stream-indicator');
      if (box) box.classList.add('hidden');
    }
  };

  es.onerror = () => {
    es.close();
    _currentES = null;
    if (_streamTimer) { clearInterval(_streamTimer); _streamTimer = null; }
    const box = document.getElementById('stream-indicator');
    if (box) box.classList.add('hidden');
    const target = document.getElementById('result-area');
    if (target && !target.innerHTML.trim()) {
      target.innerHTML = "<div class='bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700'><strong>스트리밍 연결 실패</strong><p class='mt-1 text-xs'>서버 연결을 확인하세요.</p></div>";
    }
  };
}

// ── 초기화 ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // 샘플 버튼: 선택된 pipeline에 맞춰 URL 갱신
  document.querySelectorAll('[data-qid]').forEach(btn => {
    btn.addEventListener('click', () => {
      const qid = btn.dataset.qid;
      btn.setAttribute('hx-get', getSampleUrl(qid));
      htmx.process(btn);
    });
  });

  document.querySelectorAll('input[name="pipeline"]').forEach(radio => {
    radio.addEventListener('change', () => {
      document.querySelectorAll('[data-qid]').forEach(btn => {
        const qid = btn.dataset.qid;
        btn.setAttribute('hx-get', getSampleUrl(qid));
        htmx.process(btn);
      });
      // hidden input 동기화
      const hidden = document.getElementById('pipeline-hidden');
      if (hidden) hidden.value = radio.value;
    });
  });

  // 자유 입력 폼: HTMX 대신 SSE로 가로채기
  const form = document.getElementById('query-form');
  if (form) {
    form.addEventListener('htmx:beforeRequest', (e) => {
      // HTMX 요청 차단
      e.preventDefault();
    });
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      const fd = new FormData(form);
      const question = (fd.get('question') || '').toString().trim();
      const pipeline = (fd.get('pipeline') || 'crag').toString();
      if (!question) return;
      startStreamQuery(question, pipeline);
    }, true);
  }
});
