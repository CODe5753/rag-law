"""Bounded real encoding and Qdrant ID-only reads. No generation or index writes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--embedding-origin',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.environ['EMBEDDING_BACKEND']='http';os.environ['EMBEDDING_BASE_URL']=a.embedding_origin;os.environ['EMBEDDING_TIMEOUT_SECONDS']='30'
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import embedding_backend as backend

def request(url,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=30) as r:return json.load(r)
assert backend.http_backend_ready()
queries=['개인정보 수집 동의와 목적 제한','근로시간과 휴게시간 기준','저작권 보호기간']
collections=['legal_statutes','legal_interpretation','case_law'];metadata={};results=[]
for name in collections:
 d=request('http://127.0.0.1:30633/collections/'+name)['result'];v=d['config']['params']['vectors'];assert v['size']==1024 and v['distance']=='Cosine'
 metadata[name]={'count':d.get('points_count'),'vectors':v}
for q in queries:
 candidate=backend.encode_query(q)
 reference=request('http://127.0.0.1:11434/api/embed',{'model':'bge-m3','input':[q]})['embeddings'][0]
 cosine=sum(x*y for x,y in zip(candidate,reference))/math.sqrt(sum(x*x for x in candidate)*sum(x*x for x in reference));assert cosine>.999
 rows=[]
 for name in collections:
  answers=[]
  for vector in [reference,candidate]:
   d=request('http://127.0.0.1:30633/collections/'+name+'/points/query',{'query':vector,'limit':5,'with_payload':False,'with_vector':False})['result']['points']
   answers.append([str(x['id']) for x in d])
  rows.append({'collection':name,'top5_ids_same_order':answers[0]==answers[1],'id_overlap':len(set(answers[0])&set(answers[1])),'ids_sha256':[[hashlib.sha256(x.encode()).hexdigest() for x in ids] for ids in answers]})
 results.append({'query_sha256':hashlib.sha256(q.encode()).hexdigest(),'cosine':cosine,'retrieval':rows})
result={'status':'real_candidate_encoder_and_read_only_qdrant_compared','collections':metadata,'queries':results,'index_mutation':False,'generation_calls':0,'production_deployed':False,'limits':'No payload reads; top-5 ID comparison on 3 public queries is not full retrieval/RAG correctness.'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
