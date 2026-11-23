# SARAS AI frontend

React 18 + Vite + TypeScript single-page app for **SARAS AI** (Smart Assessment and Recognition Automation System).
There is no UI kit and no chart library. The only extra dependencies are `@fontsource-variable/inter` (self-hosted font, so it works offline) and `lucide-react` (icons).

## Development

```bash
# 1. backend on :8000 (from the repo root)
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --port 8000
# 2. frontend on :5173 (proxies /api -> http://127.0.0.1:8000)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Stop both servers when you are done; the target laptop has very little free RAM.

Scripts: `npm run dev`, `npm run build` (`tsc --noEmit` + `vite build`), `npm run preview`, `npm run typecheck`.

## Production

```bash
cd frontend && npm run build
```

This produces `frontend/dist/`. When that folder exists, FastAPI serves it at `/` with an SPA fallback (non-`/api` paths return `index.html`), so only the Python process needs to run.

## Structure

```
src/
  api/        types.ts (contract types), client.ts (fetch wrapper, auth refresh, ApiError, upload, blob, CSV), endpoints.ts (one function per route)
  auth/       AuthContext (access token in memory, refresh token in localStorage, session restore on load), RequireAuth
  theme/      ThemeProvider (light | dark | system)
  components/ design system (see src/styles/README.md)
  layout/     AppShell (sidebar, blurred top bar, breadcrumbs, user menu) + ShellContext
  hooks/      useAsync, usePolling, useDelayedFlag, useDocumentTitle, useLocalStorage
  pages/      Login/Register, Dashboard, ExamLayout + Rubric, Upload, Status, Review (queue + placeholder), Results, Analytics, RegionEditor (placeholder)
  styles/     tokens.css, base.css, components.css, shell.css, pages.css
```

Routes: `/login`, `/register`, `/`, `/exams/:examId/{rubric|upload|status|review|results|analytics}`, `/exams/:examId/review/:submissionId`, `/exams/:examId/regions/:submissionId`.

## Theming

- All colours are CSS custom properties in `src/styles/tokens.css`: `:root` is the light theme and `[data-theme="dark"]` (on `<html>`) is the dark theme. Components only use semantic tokens such as `--surface`, `--text-muted`, `--accent` and `--success-soft`. Never hard-code a colour anywhere else.
- `ThemeProvider` stores the choice in `localStorage["saras-theme"]` (default `system`, which follows `prefers-color-scheme` live). It sets `data-theme`, `color-scheme` and `<meta name="theme-color">`.
- An inline script in `index.html` applies the theme before first paint, so there is no flash.
- Users can switch with the top-bar sun/moon button (cycles), the Light/Dark/System control in the user menu, or the same control in the login page corner.
- Key text/background pairs were checked for WCAG AA contrast in both themes (ratios are noted in `tokens.css`). Motion is disabled under `prefers-reduced-motion`.

See `src/styles/README.md` for the token and component reference.
