# Reddit-Topic-Analysis

## Local app

Run the localhost server from the repository root:

```powershell
python scripts/local_app_server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The local server:

- serves the frontend in `app/`
- exposes `/api/query` for routed RAG QA
- exposes `/api/status` for pipeline readiness
- exposes UI-triggerable actions for:
  - topic analysis
  - stance preview
  - app bundle rebuild
