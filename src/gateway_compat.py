"""Small Gateway-only embedding facade for legacy CLI/index/evaluation callers."""
from model_gateway import encode_batch

class GatewayEmbeddingModel:
    def get_sentence_embedding_dimension(self): return 1024

    def encode(self, texts, *, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, **kwargs):
        if kwargs or not convert_to_numpy:
            raise ValueError('Unsupported Gateway embedding output option')
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=128:
            raise ValueError('batch_size must be within1..128')
        single=isinstance(texts,str)
        items=[texts] if single else list(texts)
        vectors=[]
        for start in range(0,len(items),batch_size):
            vectors.extend(encode_batch(items[start:start+batch_size]))
        import numpy as np
        result=np.asarray(vectors,dtype=np.float32).reshape((-1,1024))
        return result[0] if single else result
