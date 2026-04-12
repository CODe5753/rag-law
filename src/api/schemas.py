from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel


class RetrievedDoc(BaseModel):
    law_name: str
    article_num: str
    text: str
    reranker_score: Optional[float] = None
    is_relevant: Optional[bool] = None   # CRAG grade 결과 (qdrant는 None)


class GradingSummary(BaseModel):
    total_docs: int
    relevant_docs: int
    threshold: float
    graded_by: str   # "reranker" | "llm" | "mixed"


class PipelineTrace(BaseModel):
    nodes_executed: list[str]
    decision: Literal["generate", "fallback"]
    grading_summary: Optional[GradingSummary] = None   # CRAG only


class HardwareInfo(BaseModel):
    model: str = "exaone3.5:7.8b"
    device: str = "Mac Mini M2 16GB"
    inference: str = "Ollama (local)"


class QueryRequest(BaseModel):
    question: str
    pipeline: Literal["crag", "qdrant"] = "crag"


class QueryResponse(BaseModel):
    question: str
    answer: str
    pipeline: str
    latency_sec: float
    cached: bool = False
    retrieved_docs: list[RetrievedDoc]
    pipeline_trace: PipelineTrace
    hardware_info: HardwareInfo = HardwareInfo()
