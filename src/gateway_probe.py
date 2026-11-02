"""Executable acceptance against installed business callers, synthetic retrieval only.
Live default: python src/gateway_probe.py
Offline test: python src/gateway_probe.py --simulated
Does not access collections, sessions, caches, ingestion, Gemini or Tavily networks.
"""
import argparse
import contextlib
import io
import json
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

# Do not let application dotenv loading read credentials from the developer tree.
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ.setdefault('EMBEDDING_BACKEND', 'ollama')
os.environ['BM25_DISABLED'] = 'true'

QUESTION = '합성 테스트 규정에서 파란 상자의 표시는 무엇인가요?'
DOC = {'chunk_id': 'synthetic-1', 'law_name': '합성테스트규정', 'article_num': '제1조',
       'text': '합성 테스트 전용 문서입니다. 파란 상자의 표시는 별입니다.', 'source': 'law'}
HISTORY = [{'role': 'user', 'content': '합성 테스트의 파란 상자를 이야기합니다.'}]


def run(simulated=False):
    import ai_gateway
    import qdrant_rag as qr
    import crag_rag as cr
    from api import pipeline_runner as runner
    from api.schemas import QueryResponse

    audit = []
    class Audit(logging.Handler):
        def emit(self, record):
            if record.msg.startswith('ai_gateway '):
                audit.append({'task': record.args[0], 'id': record.args[1]})
    handler = Audit()
    ai_gateway.logger.addHandler(handler)
    ai_gateway.logger.setLevel(logging.INFO)
    def fake(url, *, token, payload, timeout):
        assert set(payload) <= {'project', 'task', 'messages', 'response_format'}
        assert payload['project'] == 'rag-law'
        value = '{"score":0.9}' if payload['task'] == 'crag-grade' else '합성테스트규정 제1조: 별입니다.'
        return {'status_code': 200, 'data': {'id': str(uuid4()), 'text': value}}

    with contextlib.ExitStack() as stack:
        if simulated:
            stack.enter_context(patch.dict(os.environ, {'AI_GATEWAY_TOKEN': 'synthetic-test-token'}))
            stack.enter_context(patch.object(ai_gateway, 'post_json', side_effect=fake))
        stack.enter_context(patch.object(qr, 'load_embed_model', return_value=None))
        stack.enter_context(patch.object(qr, 'hybrid_retrieve', return_value=[dict(DOC)]))
        stack.enter_context(patch.object(cr, '_get_model', return_value=None))
        stack.enter_context(patch.object(cr, '_get_chunks', return_value=[dict(DOC)]))
        stack.enter_context(patch.object(cr, '_get_bm25', return_value=(None, None)))
        with contextlib.redirect_stdout(io.StringIO()):
            for result in [runner.run_qdrant(QUESTION), runner.run_qdrant_with_history(QUESTION, HISTORY), runner.run_crag(QUESTION)]:
                checked = QueryResponse.model_validate(result.model_dump())
                assert checked.answer.strip() and not checked.cached
                assert checked.retrieved_docs[0].law_name == DOC['law_name']
                assert checked.retrieved_docs[0].article_num == DOC['article_num']
                assert checked.retrieved_docs[0].text == DOC['text']
            # Exercise the real web fallback node; only the external search is synthetic.
            stack.enter_context(patch.dict(sys.modules, {'tavily': SimpleNamespace(TavilyClient=lambda **kw:
                SimpleNamespace(search=lambda *a, **kw: {'results': [{'content': DOC['text']}]}))}))
            stack.enter_context(patch.object(cr, 'TAVILY_API_KEY', 'synthetic-unusable'))
            state = cr.fallback({'question': QUESTION, 'documents': [], 'relevant_docs': []})
            assert state['fallback_used'] and state['generation'].strip()
    assert {'rewrite', 'answer', 'crag-grade', 'crag-answer', 'crag-fallback'} == {a['task'] for a in audit}
    ai_gateway.logger.removeHandler(handler)
    return {'mode': 'simulated' if simulated else 'live-gateway/synthetic-retrieval',
            'schema': 'QueryResponse + citation fields passed', 'audit': audit,
            'legal_correctness_tested': False, 'retrieval_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulated', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.simulated), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}))
        raise SystemExit(1)
