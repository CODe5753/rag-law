import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import threading
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('embedding_backend', Path(__file__).parents[1]/'src/embedding_backend.py')
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)

class TransportTests(unittest.TestCase):
 def setUp(self):
  self.mode='valid';self.calls=0;self.active=0;self.max_active=0
  owner=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_GET(self):
    data=json.dumps({'status':'ready','dimension':1024,'normalize':True,'pooling':'cls'}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));owner.calls+=1
    assert body['model']=='bge-m3' and len(body['input'])==1
    if owner.mode=='error':self.send_response(503);self.end_headers();return
    if owner.mode in {'batch', 'batch_fail', 'batch_slow'}:
     index=int(body['input'][0].split('-')[1])
     owner.active+=1;owner.max_active=max(owner.max_active,owner.active)
     if owner.mode=='batch_slow':time.sleep(.04)
     owner.active-=1
     if owner.mode=='batch_fail' and index==17:self.send_response(503);self.end_headers();return
     v=[0.0]*1024;v[index]=1.0
    else:v=[1.0]+[0.0]*1023
    if owner.mode=='short':v=v[:32]
    if owner.mode=='nan':v[0]=float('nan')
    if owner.mode=='norm':v[0]=2.0
    if owner.mode=='bool':v[1]=True
    data=json.dumps({'embeddings':[v]}).encode()
    if owner.mode=='large':data=b' '*262145
    self.send_response(200);self.end_headers()
    try:
     if owner.mode=='drip':
      for c in data[:10]:self.wfile.write(bytes([c]));self.wfile.flush();time.sleep(.08)
     else:self.wfile.write(data)
    except (BrokenPipeError,ConnectionResetError):pass
  self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
  self.env=patch.dict(os.environ,{'EMBEDDING_BASE_URL':f'http://127.0.0.1:{self.server.server_port}','EMBEDDING_TIMEOUT_SECONDS':'1'});self.env.start()
  self.previous=backend.EMBEDDING_BACKEND;backend.EMBEDDING_BACKEND='http'
 def tearDown(self):
  self.env.stop();backend.EMBEDDING_BACKEND=self.previous;self.server.shutdown();self.server.server_close();self.thread.join()
 def test_real_http_batch_and_health(self):
  result=backend.encode_batch(['공개 질의','Public query']);self.assertEqual(len(result),2);self.assertEqual(self.calls,2);self.assertTrue(backend.is_remote_backend());self.assertFalse(backend.is_ollama_backend());self.assertTrue(backend.http_backend_ready())
 def test_invalid_vectors_errors_and_no_fallback(self):
  for mode in ['short','nan','norm','bool','large','error']:
   self.mode=mode
   with self.subTest(mode=mode),patch.object(backend,'_encode_local',side_effect=AssertionError('fallback forbidden')),self.assertRaises((ValueError,RuntimeError)):
    backend.encode_query('query')
 def test_deadline_against_dripping_body(self):
  self.mode='drip';os.environ['EMBEDDING_TIMEOUT_SECONDS']='0.15';start=time.monotonic()
  with self.assertRaises((RuntimeError,TimeoutError)):backend.encode_query('query')
  self.assertLess(time.monotonic()-start,.5)
 def test_invalid_config_input_and_unknown_backend(self):
  for value in ['','file:///tmp/model','http://user:pass@localhost','http://localhost/path']:
   os.environ['EMBEDDING_BASE_URL']=value
   with self.assertRaises(ValueError):backend.encode_query('query')
  with self.assertRaises(ValueError):backend.encode_batch([''])
  backend.EMBEDDING_BACKEND='typo'
  with self.assertRaises(ValueError):backend.encode_query('query')
 def test_ingestion_32_outputs_preserve_order(self):
  self.mode='batch';os.environ['EMBEDDING_TIMEOUT_SECONDS']='3'
  result=backend.encode_batch([f'item-{i}' for i in range(32)])
  self.assertEqual(len(result),32);self.assertEqual(self.calls,32);self.assertEqual(self.max_active,1)
  self.assertEqual([max(range(1024),key=v.__getitem__) for v in result],list(range(32)))
 def test_ingestion_partial_failure_never_returns_partial_batch(self):
  self.mode='batch_fail';os.environ['EMBEDDING_TIMEOUT_SECONDS']='3'
  with patch.object(backend,'_encode_local',side_effect=AssertionError('no fallback')),self.assertRaises(RuntimeError):
   backend.encode_batch([f'item-{i}' for i in range(32)])
  self.assertEqual(self.calls,18);self.assertEqual(self.max_active,1)
 def test_ingestion_total_deadline_not_reset_per_item(self):
  self.mode='batch_slow';os.environ['EMBEDDING_TIMEOUT_SECONDS']='0.5';started=time.monotonic()
  with self.assertRaises((RuntimeError,TimeoutError)):
   backend.encode_batch([f'item-{i}' for i in range(32)])
  self.assertLess(time.monotonic()-started,.85);self.assertLess(self.calls,32)
 def test_blocked_dns_is_killed_and_reaped(self):
  with tempfile.TemporaryDirectory() as folder:
   marker=Path(folder)/'dns-worker.pid'
   source="import socket,time,os\n_original=socket.getaddrinfo\ndef blocked(*args,**kwargs):\n open("+repr(str(marker))+",'w').write(str(os.getpid()))\n time.sleep(5)\n return _original(*args,**kwargs)\nsocket.getaddrinfo=blocked\n"
   (Path(folder)/'sitecustomize.py').write_text(source)
   with patch.dict(os.environ,{'PYTHONPATH':folder,'EMBEDDING_TIMEOUT_SECONDS':'0.5'}):
    started=time.monotonic()
    with self.assertRaises(TimeoutError):backend.encode_query('query')
    self.assertLess(time.monotonic()-started,.85)
   self.assertTrue(marker.exists(),'DNS fault injection was reached')
   with self.assertRaises(ProcessLookupError):os.kill(int(marker.read_text()),0)
 def test_retrieval_callers_use_remote_gate(self):
  env=dict(os.environ,EMBEDDING_BACKEND='http',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(Path(__file__).parents[1]/'src'))
  code="import qdrant_rag, crag_rag; assert qdrant_rag.SentenceTransformer is None; assert crag_rag.SentenceTransformer is None; assert qdrant_rag.load_embed_model() is None; assert crag_rag._get_model() is None"
  subprocess.run([sys.executable,'-B','-c',code],env=env,check=True,capture_output=True,timeout=30)

if __name__=='__main__':unittest.main()
