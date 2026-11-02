"""Process-local AutoRAG adapters. Register before loading its evaluator/config."""
from llama_index.core.llms import CustomLLM,CompletionResponse,LLMMetadata
from llama_index.core.base.embeddings.base import BaseEmbedding
from ai_gateway import generate
from gateway_compat import GatewayEmbeddingModel
from model_gateway import rerank

class GatewayLlamaLLM(CustomLLM):
    @property
    def metadata(self):return LLMMetadata(context_window=32768,num_output=4096,model_name='gateway-configured')
    def complete(self,prompt,formatted=False,**kwargs):return CompletionResponse(text=generate('evaluation',prompt))
    def stream_complete(self,prompt,formatted=False,**kwargs):
        response=self.complete(prompt,formatted=formatted,**kwargs)
        yield CompletionResponse(text=response.text,delta=response.text)

class GatewayLlamaEmbedding(BaseEmbedding):
    def _get_query_embedding(self,query):return GatewayEmbeddingModel().encode(query).tolist()
    def _get_text_embedding(self,text):return self._get_query_embedding(text)
    def _get_text_embeddings(self,texts):return GatewayEmbeddingModel().encode(texts).tolist()
    async def _aget_query_embedding(self,query):return self._get_query_embedding(query)
    async def _aget_text_embeddings(self,texts):return self._get_text_embeddings(texts)


def register():
    import autorag
    import autorag.support as support
    from autorag.embedding.base import embedding_models
    from autorag.nodes.passagereranker.base import BasePassageReranker
    from autorag.utils import result_to_dataframe
    class GatewayPassageReranker(BasePassageReranker):
        @result_to_dataframe(['retrieved_contents','retrieved_ids','retrieve_scores'])
        def pure(self,previous_result,*args,**kwargs):
            queries,contents,_,ids=self.cast_to_run(previous_result)
            return self._pure(queries,contents,ids,kwargs['top_k'])
        def _pure(self,queries,contents,ids,top_k):
            result_contents=[];result_ids=[];result_scores=[]
            if not len(queries)==len(contents)==len(ids):raise ValueError('Rerank batch alignment mismatch')
            for query,docs,doc_ids in zip(queries,contents,ids):
                if len(docs)!=len(doc_ids):raise ValueError('Rerank document alignment mismatch')
                rows=rerank(query,docs,top_n=min(top_k,len(docs))) if docs else []
                result_contents.append([docs[r['index']] for r in rows]);result_ids.append([doc_ids[r['index']] for r in rows]);result_scores.append([r['relevance_score'] for r in rows])
            return result_contents,result_ids,result_scores
    autorag.generator_models['rag_gateway']=GatewayLlamaLLM
    embedding_models['rag_gateway']=autorag.LazyInit(GatewayLlamaEmbedding)
    if not getattr(support.dynamically_find_function,'rag_gateway_adapter',False):
        original=support.dynamically_find_function
        def lookup(key,target_dict):
            return GatewayPassageReranker if key=='rag_gateway_reranker' else original(key,target_dict)
        lookup.rag_gateway_adapter=True
        support.dynamically_find_function=lookup
    return GatewayPassageReranker
