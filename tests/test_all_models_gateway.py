import ast,asyncio,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import ai_gateway as gw
from api import pipeline_runner as runner

class AllModelCallers(unittest.TestCase):
 def test_intake_gateway_schema_and_errors(self):
  for output,expected in [('SEARCH: 요약',{'action':'search','summary':'요약'}),('시점을 알려주시겠어요?',{'action':'ask','reply':'시점을 알려주시겠어요?'})]:
   with patch.object(gw,'generate',return_value=output) as call:self.assertEqual(runner.intake_classify('질문',[]),expected)
   self.assertEqual(call.call_args.args[0],'intake-classify')
  with patch.object(gw,'generate',side_effect=gw.GatewayError('503')),self.assertRaises(gw.GatewayError):runner.intake_classify('질문',[])
 def test_routing_schema_and_explicit_failures(self):
  history=[{'role':'user','content':'이전 질문'}]
  with patch.object(gw,'generate',return_value='GENERAL') as call:self.assertEqual(runner.route_query('질문',history),'general')
  self.assertEqual(call.call_args.args[0],'query-intent')
  for output in ['GENERAL or RAG','unknown','']:
   with patch.object(gw,'generate',return_value=output),self.assertRaises(gw.GatewayError):runner.route_query('질문',history)
  with patch.object(gw,'generate') as call:self.assertEqual(runner.route_query('질문',[]),'rag');call.assert_not_called()
 def test_general_answer_preserves_history_and_failures(self):
  history=[{'role':'user','content':'계약 질문'}]
  with patch.dict(os.environ,{'GEMINI_API_KEY':'unused-synthetic'}),patch.object(gw,'generate',return_value='답변') as call:self.assertEqual(runner.run_general_answer('후속',history),'답변')
  self.assertEqual(call.call_args.args[0],'general-answer');self.assertEqual(call.call_args.kwargs['history'],history);self.assertIn('변호사와 상담',call.call_args.args[1])
  with patch.object(gw,'generate',side_effect=gw.GatewayError('429')),self.assertRaises(gw.GatewayError):runner.run_general_answer('질문',history)
 def test_ready_uses_gateway_only(self):
  from api.routes import ready
  import model_gateway
  for ok,code in [(True,200),(False,503)]:
   with patch.object(model_gateway,'ready',return_value=ok) as call:
    response=asyncio.run(ready());self.assertEqual(response.status_code,code);self.assertEqual(json.loads(response.body)['gateway_connected'],ok)
   self.assertEqual(set(call.call_args.args[0]),{'answer','rewrite','crag-grade','crag-answer','crag-fallback','intake-classify','query-intent','general-answer','embedding','rerank'})
 def test_three_operation_native_probe_simulated(self):
  from gateway_all_models_probe import run
  result=run(True);self.assertEqual(result['model_calls'],3);self.assertEqual({x['task'] for x in result['audit']},{'answer','embedding','rerank'})
 def test_no_direct_model_clients_in_published_python(self):
  forbidden=['sentence_transformers','ollama','google.genai','langchain_ollama','flashrank','FlagEmbedding','langchain_community.chat_models','langchain_community.embeddings']
  for path in (Path(__file__).parents[1]/'src').rglob('*.py'):
   tree=ast.parse(path.read_text())
   for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom):self.assertFalse(any((node.module or '').startswith(n) for n in forbidden),str(path))
    if isinstance(node,ast.Import):self.assertFalse(any(a.name.startswith(n) for a in node.names for n in forbidden),str(path))
    if isinstance(node,ast.Constant) and isinstance(node.value,str):
     self.assertNotIn('http://localhost:11434',node.value,str(path));self.assertNotIn('http://localhost:11435',node.value,str(path))
 def test_cli_embedding_batches_preserve_shapes(self):
  import gateway_compat as compat
  def fake(texts):return [[1.0]+[0.0]*1023 for t in texts]
  with patch.object(compat,'encode_batch',side_effect=fake) as call:result=compat.GatewayEmbeddingModel().encode(['doc']*65,batch_size=32)
  self.assertEqual(result.shape,(65,1024));self.assertEqual([len(x.args[0]) for x in call.call_args_list],[32,32,1])
  with patch.object(compat,'encode_batch',side_effect=fake):self.assertEqual(compat.GatewayEmbeddingModel().encode('single').shape,(1024,))
 def test_langchain_wrapper_uses_gateway(self):
  import gateway_langchain as lc
  from langchain_core.messages import HumanMessage,SystemMessage
  with patch.object(lc,'generate',return_value='답변') as call:result=lc.GatewayChatModel(task='evaluation').invoke([SystemMessage(content='규칙'),HumanMessage(content='질문')])
  self.assertEqual(result.content,'답변');self.assertEqual(call.call_args.args[0],'evaluation');self.assertIn('규칙',call.call_args.args[1]);self.assertIn('질문',call.call_args.args[1])
 def test_autorag_adapter_registration_and_index_mapping(self):
  import gateway_autorag as a
  R=a.register()
  import autorag.support
  self.assertEqual(autorag.support.get_support_modules('rag_gateway_reranker').__name__,R.__name__)
  with patch.object(a,'generate',return_value='답변') as call:self.assertEqual(a.GatewayLlamaLLM().complete('질문').text,'답변')
  self.assertEqual(call.call_args.args[0],'evaluation')
  with patch.object(a,'rerank',return_value=[{'index':1,'relevance_score':9},{'index':0,'relevance_score':-1}]):contents,ids,scores=R('')._pure(['q'],[['a','b']],[['A','B']],2)
  self.assertEqual(contents,[['b','a']]);self.assertEqual(ids,[['B','A']]);self.assertEqual(scores,[[9,-1]])
 def test_autorag_config_refuses_provider_overrides(self):
  from autorag_run import validate_gateway_config
  source=Path(__file__).parents[1]/'src/autorag_config.yaml';validate_gateway_config(source)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'bad.yaml';p.write_text(source.read_text().replace('embedding_model: rag_gateway','embedding_model: huggingface_bge_m3'))
   with self.assertRaises(ValueError):validate_gateway_config(p)

if __name__=='__main__':unittest.main()
