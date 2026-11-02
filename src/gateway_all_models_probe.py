"""Bounded3-call native consumer probe; no indexes, sessions, paid providers or downloads."""
import argparse,contextlib,io,json,logging,os,sys
from unittest.mock import patch
from uuid import uuid4
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['BM25_DISABLED']='true'

def run(simulated=False):
    import ai_gateway as gw
    import model_gateway as mg
    import embedding_backend as eb
    import qdrant_rag as qr
    from api import pipeline_runner as runner
    from api.schemas import QueryResponse
    docs=[{'chunk_id':'synthetic-1','law_name':'합성테스트규정','article_num':'제1조','text':'합성 테스트 전용 문서입니다. 파란 상자의 표시는 별입니다.','source':'law'}, {'chunk_id':'synthetic-2','law_name':'합성정원규정','article_num':'제2조','text':'정원의 나무에는 물을 줍니다.','source':'law'}]
    audit=[]
    class Audit(logging.Handler):
        def emit(self,record):
            if record.msg.startswith('ai_gateway '):audit.append({'task':record.args[0],'id':record.args[1]})
    handler=Audit();gw.logger.addHandler(handler)
    def fake(url,**kwargs):
        p=kwargs['payload'];assert p['project']=='rag-law' and 'model' not in p and 'provider' not in p
        if url.endswith('/ready'):
            data={'ready':True,'routes':[{'task':t,'operation':{'embedding':'embedding','rerank':'rerank'}.get(t,'generation'),'provider':'configured','model':'configured','enabled':True,'available':True} for t in p['tasks']]}
        else:
            data={'id':str(uuid4()),'provider':'configured','model':'configured','revision':1,'usage':{},'duration_ms':1}
            if url.endswith('/embeddings'):data.update(dimensions=1024,embeddings=[[1.0]+[0.0]*1023])
            elif url.endswith('/rerank'):data['results']=[{'index':0,'relevance_score':9.3},{'index':1,'relevance_score':-2.1}]
            else:data['text']='합성테스트규정 제1조에 따라 별입니다.'
        return {'status_code':200,'data':data}
    try:
        with contextlib.ExitStack() as stack:
            if simulated:
                stack.enter_context(patch.dict(os.environ,{'AI_GATEWAY_TOKEN':'synthetic'}))
                stack.enter_context(patch.object(mg,'post_json',side_effect=fake))
                stack.enter_context(patch.object(gw,'post_json',side_effect=fake))
            assert mg.ready(['answer','embedding','rerank'])
            vector=eb.encode_query('합성 규정의 파란 상자 표시')
            assert len(vector)==1024
            ranked=qr.rerank('파란 상자의 표시는 무엇인가요?',docs,top_k=2)
            assert len(ranked)==2 and {d['chunk_id'] for d in ranked}=={d['chunk_id'] for d in docs}
            stack.enter_context(patch.object(qr,'hybrid_retrieve',return_value=ranked))
            with contextlib.redirect_stdout(io.StringIO()):result=runner.run_qdrant('합성 테스트 규정에서 파란 상자의 표시는 무엇인가요?')
            checked=QueryResponse.model_validate(result.model_dump())
            assert not checked.cached and checked.answer.strip() and len(checked.retrieved_docs)==2
            assert {d.text for d in checked.retrieved_docs}=={d['text'] for d in docs}
        assert len(audit)==3 and {r['task'] for r in audit}=={'embedding','rerank','answer'}
        return {'mode':'simulated' if simulated else 'live_gateway_synthetic_retrieval','model_calls':3,'audit':audit,'embedding_dimensions':1024,'rerank_index_schema':'passed','generation_schema':'QueryResponse/citations passed','DB_index_mutations':False,'legal_correctness_tested':False}
    finally:gw.logger.removeHandler(handler)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--simulated',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.simulated),ensure_ascii=False))
