"""Bounded nonstreaming generation; retrieval and embeddings remain separate."""
import json
import logging
import os
from pathlib import Path
from uuid import UUID

from gateway_transport import post_json, validate_timeout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


class GatewayError(RuntimeError):
    pass


def configured():
    path = os.getenv('AI_GATEWAY_TOKEN_FILE')
    return bool(os.getenv('AI_GATEWAY_TOKEN') or (path and Path(path).is_file()))


def credential():
    token = os.getenv('AI_GATEWAY_TOKEN')
    if not token:
        try:
            token = Path(os.environ['AI_GATEWAY_TOKEN_FILE']).read_text().strip()
        except (KeyError, OSError):
            raise GatewayError('AI Gateway credential not configured') from None
    if not token:
        raise GatewayError('AI Gateway credential not configured')
    return token


def generate(task, prompt, *, history=None, json_mode=False, timeout=120):
    token = credential()
    messages = [{'role': h['role'], 'content': h['content']} for h in (history or [])[-6:]]
    messages.append({'role': 'user', 'content': prompt})
    payload = {'project': 'rag-law', 'task': task, 'messages': messages}
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    try:
        response = post_json(os.getenv('AI_GATEWAY_URL', 'http://100.110.193.60:3000').rstrip('/') + '/api/ai/infer',
                             token=token, payload=payload, timeout=validate_timeout(timeout))
        if response['status_code'] != 200:
            raise GatewayError(f"AI Gateway HTTP {response['status_code']}")
        data = response['data']
        request_id = str(UUID(data['id']))
        if not isinstance(data['text'], str) or not data['text'].strip():
            raise ValueError()
    except GatewayError:
        raise
    except Exception:
        raise GatewayError('AI Gateway transport or response failure') from None
    # Audit UUID survives without changing public response/citation schemas.
    logger.info('ai_gateway project=rag-law task=%s id=%s', task, request_id)
    return data['text'].strip()


def grade(prompt):
    raw = generate('crag-grade', prompt + '\nJSON 객체 {"score": 0.0}만 출력하세요.', json_mode=True)
    try:
        score = json.loads(raw)['score']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError()
        return float(score)
    except (ValueError, KeyError, TypeError):
        raise GatewayError('AI Gateway invalid grading schema') from None
