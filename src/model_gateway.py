"""All embedding/reranking requests use authenticated Gateway operations only."""
import math
import json
import os
import time
from uuid import UUID
from ai_gateway import GatewayError, credential, logger
from gateway_transport import post_json


def _call(operation, task, fields, timeout=120):
    if not 1 <= timeout <= 300:
        raise GatewayError('Gateway operation deadline exhausted')
    if len(json.dumps(fields).encode()) > 1024*1024:
        raise ValueError('Gateway model input is too large')
    response = post_json(os.getenv('AI_GATEWAY_URL','http://100.110.193.60:3000').rstrip('/')+'/api/ai/'+operation,
                         token=credential(), payload={'project':'rag-law','task':task,**fields}, timeout=timeout)
    if response.get('status_code') != 200:
        raise GatewayError(f"AI Gateway HTTP {response.get('status_code')}")
    data=response.get('data')
    try:
        request_id=str(UUID(data['id']))
        if not all(isinstance(data[k],str) and data[k].strip() for k in ['provider','model']):raise ValueError('metadata')
        if not (isinstance(data['revision'],int) and not isinstance(data['revision'],bool) and data['revision']>0):raise ValueError('metadata')
        if not (isinstance(data['duration_ms'],(int,float)) and not isinstance(data['duration_ms'],bool) and math.isfinite(data['duration_ms']) and data['duration_ms']>=0):raise ValueError('metadata')
        if data['usage'] is not None and not isinstance(data['usage'],dict):raise ValueError('metadata')
    except (ValueError,KeyError,TypeError):
        raise GatewayError('AI Gateway invalid operation metadata') from None
    logger.info('ai_gateway project=rag-law task=%s id=%s',task,request_id)
    return data


def encode_batch(texts):
    if not isinstance(texts,list) or len(texts)>128 or any(not isinstance(t,str) or not t.strip() for t in texts):
        raise ValueError('Expected at most128 nonempty texts')
    deadline=time.monotonic()+120
    vectors=[]
    for text in texts:
        data=_call('embeddings','embedding',{'input':[text]},timeout=deadline-time.monotonic())
        values=data.get('embeddings')
        if not isinstance(data.get('dimensions'),int) or data.get('dimensions')!=1024 or isinstance(data.get('dimensions'),bool) or not isinstance(values,list) or len(values)!=1:
            raise GatewayError('AI Gateway embedding count/dimension mismatch')
        v=values[0]
        if not isinstance(v,list) or len(v)!=1024 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v):
            raise GatewayError('AI Gateway invalid embedding vector')
        if abs(math.sqrt(sum(x*x for x in v))-1)>.01:
            raise GatewayError('AI Gateway embedding is not unit normalized')
        vectors.append(v)
    return vectors


def rerank(query,documents,top_n=None):
    if not isinstance(query,str) or not query.strip() or not isinstance(documents,list) or len(documents)>128 or any(not isinstance(d,str) or not d.strip() for d in documents):
        raise ValueError('Invalid reranking inputs')
    if not documents:return []
    count=len(documents) if top_n is None else top_n
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=len(documents):raise ValueError('Invalid top_n')
    fields={'query':query,'documents':documents}
    if top_n is not None:fields['top_n']=count
    data=_call('rerank','rerank',fields)
    rows=data.get('results')
    if not isinstance(rows,list) or len(rows)!=count:raise GatewayError('AI Gateway rerank count mismatch')
    seen=set();last=math.inf;last_index=-1
    for row in rows:
        if not isinstance(row,dict):raise GatewayError('AI Gateway invalid rerank row')
        index=row.get('index');score=row.get('relevance_score')
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(documents) or index in seen:raise GatewayError('AI Gateway invalid rerank index')
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or score>last:raise GatewayError('AI Gateway invalid rerank score/order')
        if score==last and index<last_index:raise GatewayError('AI Gateway unstable rerank tie order')
        seen.add(index);last=score;last_index=index
    return rows


def ready(tasks):
    """Gateway metadata readiness only; never contact a model backend directly."""
    try:
        response=post_json(os.getenv('AI_GATEWAY_URL','http://100.110.193.60:3000').rstrip('/')+'/api/ai/ready',token=credential(),payload={'project':'rag-law','tasks':tasks},timeout=3)
        data=response.get('data')
        if response.get('status_code')!=200 or not isinstance(data,dict) or data.get('ready') is not True:return False
        rows=data.get('routes')
        if not isinstance(rows,list) or len(rows)!=len(tasks):return False
        return {r.get('task') for r in rows}==set(tasks) and all(
            r.get('enabled') is True and r.get('available') is True
            and r.get('operation')==({'embedding':'embedding','rerank':'rerank'}.get(r.get('task'),'generation'))
            and all(isinstance(r.get(k),str) and r[k].strip() for k in ['provider','model'])
            for r in rows)
    except Exception:return False
