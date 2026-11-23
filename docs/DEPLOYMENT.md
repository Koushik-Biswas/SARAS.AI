# Deploying SARAS AI

SARAS AI is one FastAPI process that serves the API, the built React app and a background
worker that processes answer sheets. It needs a **long-running server with persistent disk**
(uploaded sheets, page images, the database). Pick one of the options below.

| Option | Cost | Always on | HTTPS | Best for |
|---|---|---|---|---|
| A. Linux VM + Docker + Caddy | VM cost (free with Azure for Students credit) | yes | automatic | real use |
| B. Your laptop + Cloudflare quick tunnel | free | only while the laptop runs | automatic | demos, low RAM |
| C. Frontend on Vercel + backend from A or B | free (Vercel Hobby) | as the backend | automatic | a Vercel URL |

Vercel alone cannot run SARAS AI: its functions are short-lived (no background worker), its
disk is wiped between requests (uploads and the database would vanish), and the backend's
libraries exceed its function size limit. Option C hosts only the web app on Vercel.

---

## Before any deployment: required settings

Set these as environment variables (or in `backend/.env` on the laptop; never commit it):

| Variable | Value |
|---|---|
| `SARAS_ENV` | `prod` (the app refuses to start in prod with the development JWT secret) |
| `SARAS_JWT_SECRET` | a long random string: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `SARAS_CORS_ORIGINS` | JSON list of browser origins that call the API from another domain, e.g. `["https://saras.vercel.app"]` (not needed when the app serves its own UI) |
| `SARAS_AI_GRADING`, `SARAS_AI_CONSENT`, `SARAS_GEMINI_CONSENT`, `SARAS_GROQ_API_KEY`, `SARAS_GEMINI_API_KEY` | only to enable AI reading and marking (see `backend/.env.example`) |
| `SARAS_REVIEW_REQUIRED` | `true` (default) keeps every uncertain AI mark in the teacher's review queue |

Free-tier AI limits: Groq allows about 200K tokens per day per model. A busy day of marking
can use it up; the app then falls back to the next model in `SARAS_AI_GRADING_PROVIDERS`
and, if none answers, to the offline checker / teacher review. Never a silent zero.

---

## A. Linux VM with Docker and HTTPS (recommended)

Needs a VM with 2+ vCPUs and 4+ GB RAM (8 GB with the PostgreSQL profile), Docker, a DNS
name pointing at the VM, and ports 80/443 open.

```bash
git clone https://github.com/Koushik-Biswas/SARAS.AI.git && cd SARAS.AI
cat > .env <<'EOF'
SARAS_JWT_SECRET=<long random string>
SARAS_DB_PASSWORD=<database password, no URL special characters>
SARAS_DOMAIN=saras.example.edu
SARAS_APP_CPUS=2
EOF
docker compose -f docker-compose.yml -f deploy/docker-compose.https.yml up -d --build
```

Caddy obtains and renews the TLS certificate; the app is reachable only through Caddy.
Add AI keys and consents to the `app` service's environment (or the `.env` above with an
`env_file:` entry) when you want AI reading and marking.

Optional offline semantic checker (ONNX NLI model, ~90 MB, not in the repository):
`docker compose exec app python -m scripts.download_models`, then restart.

Without a domain (testing only): `docker compose up -d --build` serves plain HTTP on port 8000.

Single container without PostgreSQL (SQLite in the `/data` volume):

```bash
docker build -t saras-ai .
docker run -d -p 8000:8000 -v saras-data:/data -e SARAS_JWT_SECRET=<secret> saras-ai
```

The image honours `PORT` (set by most hosting platforms) and trusts `X-Forwarded-*` only from
`FORWARDED_ALLOW_IPS` (default `127.0.0.1`; set it to the proxy's address, or `*` when the
container is reachable only through that proxy).

## B. Laptop + Cloudflare quick tunnel (free demo link)

```powershell
# once: backend venv, frontend build (see README "Quick start")
powershell -ExecutionPolicy Bypass -File scripts\run_prod.ps1      # http://127.0.0.1:8000
# second terminal (install: winget install Cloudflare.cloudflared)
cloudflared tunnel --url http://127.0.0.1:8000                    # prints https://<random>.trycloudflare.com
```

The tunnel URL changes every run and stops when the laptop sleeps. `run_prod.ps1` starts
uvicorn with `--proxy-headers`, so rate limiting sees real visitor IPs through the tunnel.

## C. Web app on Vercel, backend on A or B

1. Deploy the backend (A or B) and note its public HTTPS origin, e.g. `https://api.example.edu`.
2. In Vercel: **Add New → Project → Import** `Koushik-Biswas/SARAS.AI`, set **Root Directory**
   to `frontend` (framework, build command and output come from `frontend/vercel.json`).
3. Add the environment variable `VITE_API_ORIGIN=https://api.example.edu` and deploy.
4. On the backend set `SARAS_CORS_ORIGINS=["https://<your-project>.vercel.app"]` and restart.

Sign-in uses bearer tokens (no cookies), so the cross-domain setup needs nothing else.

---

## Backups

```bash
python -m scripts.backup --keep 14          # from backend/: database + uploads -> backend/backups/*.zip
```

SQLite is copied online (safe while the app runs). With PostgreSQL also dump the database:
`docker compose exec db pg_dump -U saras -Fc saras > saras-$(date +%F).dump`.
Schedule it daily (cron / Task Scheduler) and copy the files off the machine. Backups contain
student work: keep them private.

## After deploying: checklist

- `https://<host>/health` returns `{"status":"ok"}`; `/ready` reports the database and worker.
- Response headers include `Content-Security-Policy`, `X-Frame-Options: DENY` and, over HTTPS,
  `Strict-Transport-Security`.
- Register the first teacher account, create an exam, upload a sample sheet from
  `backend/samples/sheets/` (generate with `python -m scripts.make_samples`) and watch the Status page.
- Change any demo passwords; never run `scripts.seed_demo` in production.

## Measured engine notes (2026-11-24)

Two experimental marking features are **off by default** because the dev benchmark showed
them less accurate: scheme-based marking for question-only rubrics (`SARAS_AI_DIRECT_SCHEME`)
and the selective second marking (`SARAS_AI_VERIFY`). See `backend/evaluation/reports/`.
The held-out benchmark has not been run yet (the free daily AI quota ran out).
