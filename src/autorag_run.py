"""AutoRAG entrypoint that registers Gateway-only model adapters before evaluation."""
import argparse
from pathlib import Path
import yaml

def validate_gateway_config(path):
    data=yaml.safe_load(Path(path).read_text())
    for db in data.get('vectordb',[]):
        if db.get('embedding_model')!='rag_gateway':raise ValueError('AutoRAG embeddings must use Gateway')
    for line in data.get('node_lines',[]):
        for node in line.get('nodes',[]):
            for module in node.get('modules',[]):
                if any(k in module for k in ['api_base','api_key','model','model_name']):raise ValueError('Direct model overrides are forbidden')
                if node['node_type']=='generator' and not (module.get('module_type')=='llama_index_llm' and module.get('llm')=='rag_gateway'):raise ValueError('AutoRAG generation must use Gateway')
                if node['node_type']=='passage_reranker' and module.get('module_type') not in ['pass_reranker','rag_gateway_reranker']:raise ValueError('AutoRAG reranking must use Gateway')
    return data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='src/autorag_config.yaml');parser.add_argument('--qa',default='data/autorag/qa.parquet');parser.add_argument('--corpus',default='data/autorag/corpus.parquet');parser.add_argument('--project-dir',default='data/autorag/results');args=parser.parse_args()
    validate_gateway_config(args.config)
    from gateway_autorag import register
    register()
    from autorag.evaluator import Evaluator
    Evaluator(args.qa,args.corpus,project_dir=args.project_dir).start_trial(args.config)

if __name__=='__main__':main()
