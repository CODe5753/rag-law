// 경과 시간 타이머
let _timer = null;

function startTimer() {
  const el = document.getElementById('elapsed');
  if (!el) return;
  let sec = 0;
  el.textContent = '0';
  _timer = setInterval(() => {
    sec++;
    el.textContent = sec;
  }, 1000);
}

function stopTimer() {
  if (_timer) { clearInterval(_timer); _timer = null; }
}

// HTMX 이벤트
document.addEventListener('htmx:beforeRequest', (e) => {
  if (e.detail.elt.closest('form') || e.detail.elt.dataset.htmxLoading) {
    startTimer();
  }
});

document.addEventListener('htmx:afterRequest', stopTimer);
document.addEventListener('htmx:responseError', stopTimer);

// 샘플 버튼 클릭 시 선택된 파이프라인 반영
function getSampleUrl(qid) {
  const pipeline = document.querySelector('input[name="pipeline"]:checked')?.value || 'crag';
  return `/samples/${qid}?pipeline=${pipeline}`;
}

// 샘플 버튼 동적 URL 설정
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-qid]').forEach(btn => {
    btn.addEventListener('click', () => {
      const qid = btn.dataset.qid;
      btn.setAttribute('hx-get', getSampleUrl(qid));
      htmx.process(btn);
    });
  });

  // 파이프라인 변경 시 샘플 버튼 URL 갱신
  document.querySelectorAll('input[name="pipeline"]').forEach(radio => {
    radio.addEventListener('change', () => {
      document.querySelectorAll('[data-qid]').forEach(btn => {
        const qid = btn.dataset.qid;
        btn.setAttribute('hx-get', getSampleUrl(qid));
        htmx.process(btn);
      });
    });
  });
});
