# EduGenAI — Production GenAI Platform (Week 4 Capstone)

A production-oriented starter project for an education-focused GenAI assistant. Built with FastAPI, Gemini, LangChain, ChromaDB and a lightweight HTML frontend.

## Features

- **RAG pipeline:** Upload PDF documents via API, chunk them, embed them and retrieve relevant passages from ChromaDB.
- **Streaming responses:** `POST /api/chat/stream` streams Gemini output using Server-Sent Events (SSE).
- **Conversational memory:** In-memory session history retained for a bounded number of turns.
- **Multimodal input:** `POST /api/vision` accepts an image or PDF and asks Gemini to explain it.
- **Guardrails:** Input length limits, a small demonstration blocklist, instructions to treat document content as untrusted, and domain caution for medical/legal/financial questions.
- **Authentication option:** Set `APP_ACCESS_TOKEN`; protected API routes require `Authorization: Bearer <token>`.
- **Rate limiting:** Basic per-client in-process limit. For multi-worker production, replace this with Redis/API gateway rate limiting.
- **Usage analytics:** SQLite counters for chat requests, uploads, character-count proxies and average chat latency.
- **API documentation:** Swagger UI at `/docs`, health check at `/health`.
- **Responsive UI:** Simple web client for streaming chat and image/PDF questions.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Gemini API key, and change `APP_ACCESS_TOKEN` to a long random secret. Keep `.env` private.

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 and API docs at http://127.0.0.1:8000/docs.

## API overview

- `GET /health` — health status
- `GET /api/docs` — endpoint overview
- `POST /api/upload` — multipart PDF upload and indexing
- `POST /api/chat/stream` — JSON `{ "message": "...", "session_id": null, "use_knowledge_base": true }`, SSE stream
- `POST /api/vision` — multipart `file` + `question` for image/PDF understanding
- `GET /api/analytics` — usage metrics (protected when token is configured)

When `APP_ACCESS_TOKEN` is set, protected endpoints require:
`Authorization: Bearer YOUR_TOKEN`

## Deploy

### Render / Railway (simple path)
1. Push this repository to GitHub.
2. Create a new Python web service from the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add `GOOGLE_API_KEY` and a strong `APP_ACCESS_TOKEN` as service environment variables.
6. Configure persistent storage mounted at `/app/data` and set `DATA_DIR=/app/data` if the host supports it. Without persistent storage, uploaded PDFs, ChromaDB and SQLite analytics can be lost on redeploy.
7. Test `/health`, `/docs`, PDF upload, streaming chat and analytics.

### Vercel / AWS
This app uses local disk for ChromaDB and SQLite. For reliable Vercel or multi-instance AWS deployment, replace local persistence with managed vector storage and a managed database/object store; configure streaming and request limits at the platform layer.

## Security and production-readiness notes

This is a capstone starter, not a fully audited production service. Before public deployment:
- Replace the demo blocklist with layered moderation and abuse monitoring.
- Use a proper identity provider, scoped user accounts and authorization; the shared token is only a simple access gate.
- Move rate limits to Redis/API gateway for multi-instance deployments.
- Use managed persistent storage, encrypted secrets and secure upload scanning.
- Add token-accurate billing telemetry from provider usage metadata; character counts are only rough proxies.
- Add tests, structured logs, error monitoring, retention/deletion policies and privacy notices.
- OCR is not included for scanned PDFs.
- Treat model outputs as untrusted and validate them before downstream actions.

## Architecture

Browser UI → FastAPI → guardrails/auth/rate-limit → ChromaDB retrieval → Gemini streaming response → browser.
Uploads → PDF parser → text chunks → Gemini embeddings → ChromaDB.
Image/PDF questions → Gemini multimodal endpoint.
Usage events → SQLite analytics table.

## License

Educational starter project. Add a license if you intend to distribute it.
