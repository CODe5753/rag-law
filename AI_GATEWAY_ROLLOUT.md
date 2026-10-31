# rag-law Gateway migration — 2026-10-31

Status: **ready_for_parent_verification**, not deployed or complete. COMMON.md assigns publication, credentials, first live probe and rollout to the parent. No commits, pushes, secret reads, ingestion, collection writes, reindex, service restarts or shared repository edits were performed.

## Source and preservation

Current main is `83b35db2f2d0100d4faa5f53ecb3dfe0edf1a244`, ahead 1 / behind 0 of origin/main `f7505d83e9fea543b4c6dfa3ed812925c5b4e14e`. `git fetch origin` was denied writing `.git/FETCH_HEAD`; fresh `git ls-remote` verified the same main hash and no other remote branch. Production is main via `.github/workflows/deploy.yml`; no test branch/workflow found. Ahead commit is reranker research and must not be pushed with this migration.

Exact before-edit copies and SHA256 hashes are in `.hermes/gateway-orchestration-baseline/manifest.json`. Deleted README, dirty `.gitignore`/monitor, untracked research, reranker service and ingest files are preserved. No staged index changes. CRAG's preexisting one-line correction removing the extra `chunks` argument is preserved and explicitly included as a necessary caller compatibility dependency in the promotion patch. No other preexisting changes are included.

## Implemented routes and schemas

| Actual caller | Gateway task |
|---|---|
| API `rewrite_query` with history | `rewrite` |
| Qdrant `generate`, including `generate_with_gemini` SDK/key/generation failures | `answer` |
| CRAG document grade when no reranker score exists | `crag-grade` |
| CRAG relevant-document answer | `crag-answer` |
| CRAG answer after optional web search | `crag-fallback` |

Project is `rag-law`. Parent must bind all five tasks to Codex. Payload contains only project/task/messages and grading response_format. History is preserved on Gemini failure. No provider/model/temperature/stream fields. Shared client transport was adapted from content-poster read-only reference: isolated subprocess, 120-second wall deadline per inference, 1 MiB response cap, no retries or fan-out, token through stdin, sanitized errors. Native UUID `id` is validated and logged with task, without prompts or tokens. HTTP 401/403/429/502/503/504 raise explicit errors; grading rejects malformed/non-numeric/out-of-range scores. Gateway failure is not caught as successful web fallback or rewritten original question.

`QueryResponse`, `RetrievedDoc`, grading state, source/citation fields and SSE event shapes are unchanged. Hardware labels identify route-dependent Gateway/Gemini inference. SSE remains node/final events, not token streaming; existing disconnect/background worker behavior is unchanged, and no claim of cancellation of an already running inference is made. Readiness checks credential configuration and Ollama connectivity only when embeddings use Ollama; it is explicitly not a Gateway inference-health check. A live probe is required before rollout acceptance.

Gemini primary intake, routing, general answer and history-answer paths remain as before. This scope replaces the actual Ollama production generation paths and Gemini's Ollama fallback. Offline research/evaluation generation remains inventoried in JSON; it is not part of the deployed API graph. No ingestion/evaluation pipeline was executed or changed.

## Embeddings and reranker

Local `embedding_backend.encode_query` was executed offline with cached `BAAI/bge-m3` and synthetic text: 1024 dimensions, finite values, norm 1.0. Existing environment selection correctly dispatches local sentence-transformer. Read-only collection metadata from the deployed pod reports 1024/Cosine for legal_statutes, legal_interpretation and case_law. This proves shape compatibility, not equal retrieval quality between Ollama F16 and local weights/tokenization.

Deployment remains `EMBEDDING_BACKEND=ollama`, `OLLAMA_EMBEDDING_MODEL=bge-m3`. Production requirements do not include sentence-transformers/torch; local migration needs a separately reviewed image dependency change, offline cache verification and memory check against the 3Gi limit, plus synthetic vector comparison before switching. No model download, reindex or backend change performed. Remote reranker `http://host.lima.internal:11435` is unchanged and separate from generation.

## Validation

```sh
PYTHON_DOTENV_DISABLED=1 EMBEDDING_BACKEND=ollama .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/gateway_probe.py --simulated
```

The promotion patch was also applied to an isolated origin/main source tree and all seven tests passed there without the ahead research commit or unrelated work.

Seven tests pass: explicit HTTP statuses, malformed responses/UUIDs, JSON grade schema, deadline bounds, oversized body rejection, actual business caller schemas and no duplicate Gateway attempt on no-key Gemini fallback. AST parsing of 8 modules and `git diff --check` pass. Simulated probe made six calls across all five tasks. Those UUIDs are generated test values, not evidence of live Gateway success.

The probe runs actual API pipeline functions with synthetic retrieval and actual CRAG graph execution; only document retrieval, external Tavily and Gemini failure are substituted. It asserts response and citation schemas. It never opens production sessions, fetches stored documents, sends notifications or evaluates legal correctness. It imports deployed modules directly, with dotenv disabled. Live mode never substitutes Gateway transport.

## Parent publication and deployed acceptance

Do not push this checkout's main: its unrelated ahead commit would be included. Use a clean checkout at the freshly fetched origin/main, confirm the base is still f7505d8, and apply the allowlisted patch:

```sh
git apply --check /Users/seunghyeon/dev/rag-law/.hermes/gateway-promotion.patch
git apply /Users/seunghyeon/dev/rag-law/.hermes/gateway-promotion.patch
```

Use `publish_allowlist` in JSON for scoped staging and review. The patch includes source, tests and these two reports only. Exclude `.hermes`, README deletion, `.gitignore`, monitor, ingest/research, data and the ahead commit. If origin advanced, regenerate/review against new source rather than assuming applicability. Parent coordinates commit/push after checking the narrow delta; main push builds arm64 GHCR and the workflow updates GitOps automatically, so arrange token/routes before publication.

Proposed shared manifest insertion under `homelab-gitops/rag-law/deployment.yaml` container env (parent-owned, not applied):

```yaml
- name: AI_GATEWAY_URL
  value: "http://100.110.193.60:3000"
- name: AI_GATEWAY_TOKEN
  valueFrom:
    secretKeyRef:
      name: rag-law-ai-gateway
      key: AI_GATEWAY_TOKEN
```

The secret name is a proposal, not a claim that it exists. Retain all existing embedding, reranker, Gemini and volume settings. Image observed in both manifest and cluster is `ghcr.io/code5753/rag-law:sha-f7505d8`, namespace `rag-law`, workload `rag-law-demo`; volumes `hf-model-cache` and `rag-law-sessions`. The Dockerfile copies `src/`, so the probe is included in the deployed artifact without image layout changes.

After deploying the reviewed image and provisioning routes/token, parent runs:

```sh
kubectl -n rag-law exec deployment/rag-law-demo -- python src/gateway_probe.py
```

Require exit 0, `mode=live-gateway/synthetic-retrieval`, successful schema assertion, and real Gateway UUIDs for all five tasks. Verify provider Codex through parent-owned Gateway audit for those UUIDs. The probe uses synthetic document fixtures and forced Gemini SDK failure; it does not claim real collection retrieval or legal correctness. No live deployed-caller result is available in this phase. First working live evidence is a mandatory remaining acceptance gate.

Rollback: parent restores `ghcr.io/code5753/rag-law:sha-f7505d8` in GitOps and waits for ArgoCD rollout. Existing Ollama embedding and reranker stay available; no data rollback or reindex needed.
