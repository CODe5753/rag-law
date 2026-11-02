"""LangChain interfaces with no direct provider/model client behind them."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration,ChatResult
from langchain_core.embeddings import Embeddings
from ai_gateway import generate
from gateway_compat import GatewayEmbeddingModel

class GatewayChatModel(BaseChatModel):
    task: str='answer'
    @property
    def _llm_type(self):return 'internal-gateway'
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        # Keep the complete prompt and roles rather than silently dropping eval messages.
        prompt='\n\n'.join(f'[{m.type}]\n{m.content}' for m in messages)
        text=generate(self.task,prompt)
        if stop:
            positions=[text.find(token) for token in stop if token and token in text]
            if positions:text=text[:min(positions)]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

class GatewayEmbeddings(Embeddings):
    def embed_documents(self,texts):return GatewayEmbeddingModel().encode(texts).tolist()
    def embed_query(self,text):return GatewayEmbeddingModel().encode(text).tolist()
