"""Gateway operation contracts replace the retired direct-backend transport tests."""
import ast,json,math,os,sys,threading,time,unittest,tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import model_gateway as mg
import embedding_backend as eb
import ai_gateway as gw
from gateway_transport import post_json

def metadata(**fields):
 return {'id':str(uuid4()),'provider':'bge-runtime','model':'configured-model','revision':1,'usage':{},'duration_ms':1,**fields}
def vector(index=0):
 v=[0.0]*1024;v[index]=1.0;return v

class GatewayModelTests(unittest.TestCase):
 def setUp(self):self.log=patch.object(gw.logger,'disabled',True);self.log.start();self.env=patch.dict(os.environ,{'AI_GATEWAY_TOKEN':'synthetic','AI_GATEWAY_URL':'http://gateway.invalid'});self.env.start()
 def tearDown(self):self.env.stop();self.log.stop()
 def test_32_batch_preserves_order_and_gateway_only_contract(self):
  calls=[]
  def fake(url,**kw):
   p=kw['payload'];self.assertEqual(set(p),{'project','task','input'});self.assertEqual(p['task'],'embedding');self.assertEqual(kw['token'],'synthetic');self.assertTrue(url.endswith('/api/ai/embeddings'));self.assertEqual(len(p['input']),1)
   i=int(p['input'][0]);calls.append(i);return {'status_code':200,'data':metadata(dimensions=1024,embeddings=[vector(i)])}
  with patch.object(mg,'post_json',side_effect=fake):values=eb.encode_batch([str(i) for i in range(32)])
  self.assertEqual(calls,list(range(32)));self.assertEqual([v.index(1) for v in values],list(range(32)))
 def test_batch_failure_no_partial_result_or_retry(self):
  count=0
  def fake(*a,**kw):
   nonlocal count
   count+=1
   return {'status_code':503} if count==18 else {'status_code':200,'data':metadata(dimensions=1024,embeddings=[vector()])}
  with patch.object(mg,'post_json',side_effect=fake),self.assertRaises(gw.GatewayError):eb.encode_batch(['text']*32)
  self.assertEqual(count,18)
 def test_batch_deadline_is_shared(self):
  budgets=[]
  def fake(*a,**kw):budgets.append(kw['timeout']);time.sleep(.01);return {'status_code':200,'data':metadata(dimensions=1024,embeddings=[vector()])}
  with patch.object(mg,'post_json',side_effect=fake):eb.encode_batch(['a','b','c'])
  self.assertGreater(budgets[0],budgets[1]);self.assertGreater(budgets[1],budgets[2])
 def test_bad_vectors_or_metadata_fail(self):
  for data in [metadata(dimensions=32,embeddings=[vector()]),metadata(dimensions=1024,embeddings=[vector()[:32]]),metadata(dimensions=1024,embeddings=[[float('nan')]+[0]*1023]),metadata(dimensions=1024,embeddings=[[True]+[0]*1023]),metadata(dimensions=1024,embeddings=[[2]+[0]*1023]),metadata(dimensions=1024,embeddings=[]),{'id':'bad'},metadata(revision=True,dimensions=1024,embeddings=[vector()])]:
   with self.subTest(data_keys=list(data)),patch.object(mg,'post_json',return_value={'status_code':200,'data':data}),self.assertRaises(gw.GatewayError):eb.encode_query('test')
 def test_rerank_unbounded_scores_and_index_mapping(self):
  import qdrant_rag
  rows=[{'index':1,'relevance_score':8.9},{'index':0,'relevance_score':-3.1}]
  docs=[{'text':'a','chunk_id':'A'},{'text':'b','chunk_id':'B'}]
  with patch.object(mg,'post_json',return_value={'status_code':200,'data':metadata(results=rows,usage=None)}) as call:result=qdrant_rag.rerank('question',docs,top_k=2)
  self.assertEqual([d['chunk_id'] for d in result],['B','A']);self.assertEqual([d['reranker_score'] for d in result],[8.9,-3.1]);self.assertEqual(set(call.call_args.kwargs['payload']),{'project','task','query','documents','top_n'})
 def test_bad_rerank_indexes_scores_or_order_fail(self):
  cases=[[{'index':0,'relevance_score':2},{'index':0,'relevance_score':1}],[{'index':3,'relevance_score':2},{'index':0,'relevance_score':1}],[{'index':0,'relevance_score':float('nan')},{'index':1,'relevance_score':1}],[{'index':0,'relevance_score':0},{'index':1,'relevance_score':1}],[{'index':1,'relevance_score':1},{'index':0,'relevance_score':1}],[]]
  for rows in cases:
   with patch.object(mg,'post_json',return_value={'status_code':200,'data':metadata(results=rows)}),self.assertRaises(gw.GatewayError):mg.rerank('query',['a','b'])
 def test_http_errors_no_silent_fallback(self):
  for status in [401,403,429,502,503,504]:
   with patch.object(mg,'post_json',return_value={'status_code':status}) as call,self.assertRaisesRegex(gw.GatewayError,str(status)):eb.encode_query('test')
   self.assertEqual(call.call_count,1)
 def test_ready_checks_gateway_route_metadata(self):
  rows=[{'task':'embedding','enabled':True,'available':True,'operation':'embedding','provider':'local','model':'bge-m3'}]
  with patch.object(mg,'post_json',return_value={'status_code':200,'data':{'ready':True,'routes':rows}}) as call:self.assertTrue(mg.ready(['embedding']))
  self.assertTrue(call.call_args.args[0].endswith('/api/ai/ready'));self.assertEqual(call.call_args.kwargs['payload'],{'project':'rag-law','tasks':['embedding']})
  for data in [{'ready':False,'routes':rows},{'ready':True,'routes':[]},{'ready':True,'routes':[{'task':'embedding','enabled':True,'available':False}]}]:
   with patch.object(mg,'post_json',return_value={'status_code':200,'data':data}):self.assertFalse(mg.ready(['embedding']))
 def test_subprocess_dns_deadline_kills_worker(self):
  with tempfile.TemporaryDirectory() as folder:
   marker=Path(folder)/'pid'
   (Path(folder)/'sitecustomize.py').write_text("import socket,os,time\ndef blocked(*a,**k):\n open("+repr(str(marker))+",'w').write(str(os.getpid()))\n time.sleep(10)\nsocket.getaddrinfo=blocked\n")
   with patch.dict(os.environ,{'PYTHONPATH':folder}):
    started=time.monotonic()
    with self.assertRaises(TimeoutError):post_json('http://gateway.invalid/api/ai/embeddings',token='synthetic',payload={},timeout=1)
    self.assertLess(time.monotonic()-started,1.6)
   self.assertTrue(marker.exists())
   with self.assertRaises(ProcessLookupError):os.kill(int(marker.read_text()),0)
 def test_real_http_gateway_transport_and_readiness(self):
  captured=[]
  class H(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));captured.append((self.path,body,self.headers.get('Authorization')))
    response=metadata(dimensions=1024,embeddings=[vector()]) if self.path.endswith('embeddings') else {'ready':True,'routes':[{'task':'embedding','enabled':True,'available':True,'operation':'embedding','provider':'local','model':'bge-m3'}]}
    data=json.dumps(response).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
  server=ThreadingHTTPServer(('127.0.0.1',0),H);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with patch.dict(os.environ,{'AI_GATEWAY_URL':f'http://127.0.0.1:{server.server_port}'}):self.assertEqual(len(eb.encode_query('query')),1024);self.assertTrue(mg.ready(['embedding']))
   self.assertEqual([p for p,_,_ in captured],['/api/ai/embeddings','/api/ai/ready']);self.assertTrue(all(auth=='Bearer synthetic' for _,_,auth in captured))
  finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
