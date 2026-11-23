#!/usr/bin/env node
/**
 * SARAS AI developer launcher. Run it from the repository root through npm
 * (no extra dependencies, works in PowerShell, cmd, Git Bash, macOS and Linux):
 *
 *   npm install        install the web app's dependencies (frontend/)
 *   npm run setup      create backend/.venv and install Python + web dependencies
 *   npm run dev        API on :8000 (auto-reload) + web app on :5173, one terminal
 *   npm run dev:api    API only
 *   npm run dev:web    web app only (expects the API on :8000)
 *   npm run build      build the web app into frontend/dist
 *   npm start          production-like single process: FastAPI serves frontend/dist on :8000
 *   npm test           backend test suite (extra pytest arguments go after --)
 *   npm run seed       load the demo teacher, exam and sample sheets
 *
 * Ctrl+C stops everything this launcher started, child process trees included.
 * Environment overrides: SARAS_API_HOST, SARAS_API_PORT, SARAS_WEB_PORT.
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BACKEND = path.join(ROOT, "backend");
const FRONTEND = path.join(ROOT, "frontend");
const IS_WIN = process.platform === "win32";
const VENV_PY = IS_WIN
  ? path.join(BACKEND, ".venv", "Scripts", "python.exe")
  : path.join(BACKEND, ".venv", "bin", "python");
const VITE_BIN = path.join(FRONTEND, "node_modules", "vite", "bin", "vite.js");
const API_HOST = process.env.SARAS_API_HOST || "127.0.0.1";
const API_PORT = Number(process.env.SARAS_API_PORT || 8000);
const WEB_PORT = Number(process.env.SARAS_WEB_PORT || 5173);
// Seconds after start-up at which the launcher stops itself. Used only by smoke tests.
const EXIT_AFTER = Number(process.env.SARAS_LAUNCHER_EXIT_AFTER || 0);

const useColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const paint = (code) => (text) => (useColor ? `\x1b[${code}m${text}\x1b[0m` : text);
const COLORS = { api: paint("36"), web: paint("35"), saras: paint("32"), error: paint("31"), bold: paint("1") };

function say(message, tag = "saras") {
  const color = COLORS[tag] || ((t) => t);
  console.log(`${color(`[${tag === "error" ? "saras" : tag}]`)} ${message}`);
}

function die(message) {
  console.error(`${COLORS.error("[saras]")} ${message}`);
  process.exit(1);
}

const rel = (p) => path.relative(ROOT, p) || ".";

// ---------------------------------------------------------------- processes
const children = new Set();
let stopping = false;

function childEnv() {
  const env = { ...process.env, PYTHONUNBUFFERED: "1" };
  // Children write into pipes, so they would drop colours on their own; force them on only when
  // our own output is a terminal. Any FORCE_COLOR value (even "0") turns colours ON in picocolors,
  // so when colours are off it must be removed and NO_COLOR set instead.
  if (useColor) {
    env.FORCE_COLOR = "1";
  } else {
    delete env.FORCE_COLOR;
    env.NO_COLOR = "1";
  }
  return env;
}

function startChild(tag, command, args, cwd) {
  const child = spawn(command, args, {
    cwd,
    env: childEnv(),
    // stdin is not shared: a background process group reading the terminal would be suspended on POSIX.
    stdio: ["ignore", "pipe", "pipe"],
    // Own process group on POSIX so the whole tree (e.g. uvicorn's reloader) can be stopped.
    // Not on Windows, where `detached` opens a new console window.
    detached: !IS_WIN,
    windowsHide: true,
  });
  children.add(child);
  for (const stream of [child.stdout, child.stderr]) {
    readline.createInterface({ input: stream }).on("line", (line) => console.log(`${COLORS[tag](`[${tag}]`)} ${line}`));
  }
  child.on("error", (error) => {
    say(`${tag} could not start: ${error.message}`, "error");
    stopAll(1);
  });
  child.on("exit", (code, signal) => {
    children.delete(child);
    // On Windows, Ctrl+C reaches the children too and one of them may exit before our own
    // SIGINT handler has run; wait a moment so a normal Ctrl+C is not reported as a crash.
    setTimeout(() => {
      if (!stopping) {
        say(`${tag} stopped unexpectedly (${signal || `exit code ${code}`}); stopping everything.`, "error");
        stopAll(code || 1);
      }
    }, 300);
  });
  return child;
}

function isRunning(child) {
  return child.exitCode === null && child.signalCode === null;
}

function killTree(child, force = false) {
  if (!isRunning(child)) return;
  try {
    if (IS_WIN) {
      spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
    } else {
      process.kill(-child.pid, force ? "SIGKILL" : "SIGTERM");
    }
  } catch {
    // already gone
  }
}

function stopAll(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) killTree(child);
  const deadline = Date.now() + 3000;
  const timer = setInterval(() => {
    const alive = [...children].filter(isRunning);
    if (alive.length === 0 || Date.now() > deadline) {
      for (const child of alive) killTree(child, true);
      clearInterval(timer);
      process.exit(code);
    }
  }, 100);
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => {
    if (children.size) say("Stopping…");
    stopAll(0);
  });
}

// ------------------------------------------------------------------ helpers
function portInUse(port) {
  const probe = (host) =>
    new Promise((resolve) => {
      const socket = net.connect({ host, port });
      const finish = (used) => {
        socket.destroy();
        resolve(used);
      };
      socket.setTimeout(500, () => finish(false));
      socket.once("connect", () => finish(true));
      socket.once("error", () => finish(false));
    });
  return Promise.all([probe("127.0.0.1"), probe("::1")]).then((results) => results.some(Boolean));
}

async function waitForHttp(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline && !stopping) {
    const ok = await new Promise((resolve) => {
      const request = http.get(url, (response) => {
        response.resume();
        resolve(response.statusCode < 500);
      });
      request.setTimeout(2000, () => {
        request.destroy();
        resolve(false);
      });
      request.on("error", () => resolve(false));
    });
    if (ok) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

/** Run npm synchronously in `cwd`; returns the exit code. */
function runNpm(args, cwd) {
  // Prefer the npm that launched us (no shell needed); npm is a .cmd shim on Windows otherwise.
  const npmCli = process.env.npm_execpath;
  let result;
  if (npmCli && /\.(c?js|mjs)$/.test(npmCli)) {
    result = spawnSync(process.execPath, [npmCli, ...args], { cwd, stdio: "inherit" });
  } else if (IS_WIN) {
    // Our arguments are fixed literals, so one command string through the shell is safe.
    result = spawnSync(["npm", ...args].join(" "), { cwd, stdio: "inherit", shell: true });
  } else {
    result = spawnSync("npm", args, { cwd, stdio: "inherit" });
  }
  return result.status ?? 1;
}

function requireVenv() {
  if (!existsSync(VENV_PY)) die(`Python virtual environment not found (${rel(VENV_PY)}). Run: npm run setup`);
  return VENV_PY;
}

function requireBackend() {
  if (!existsSync(path.join(BACKEND, "app", "main.py"))) {
    die("backend/app/main.py is missing, so the API cannot start yet.");
  }
}

function ensureWebDeps() {
  if (existsSync(VITE_BIN)) return;
  if (!existsSync(path.join(FRONTEND, "package.json"))) die("frontend/package.json is missing.");
  say("Installing the web app's dependencies (first run)…");
  if (runNpm(["install", "--no-audit", "--no-fund"], FRONTEND) !== 0 || !existsSync(VITE_BIN)) {
    die("npm install in frontend/ failed — see the output above.");
  }
}

function warnIfWebIncomplete() {
  if (!existsSync(path.join(FRONTEND, "src", "main.tsx"))) {
    say("frontend/src/main.tsx does not exist yet, so the page will show an error until the web app is built.", "error");
  }
}

const healthUrl = () => `http://127.0.0.1:${API_PORT}/health`;

function startApi({ reload }) {
  const args = ["-m", "uvicorn", "app.main:app", "--host", API_HOST, "--port", String(API_PORT)];
  // Watch only the application package — never .venv or data/ (too many files for a small laptop).
  if (reload) args.push("--reload", "--reload-dir", "app");
  return startChild("api", requireVenv(), args, BACKEND);
}

function startWeb() {
  return startChild("web", process.execPath, [VITE_BIN, "--port", String(WEB_PORT), "--strictPort"], FRONTEND);
}

function scheduleSmokeExit() {
  if (EXIT_AFTER > 0) {
    setTimeout(() => {
      say(`SARAS_LAUNCHER_EXIT_AFTER=${EXIT_AFTER}s reached; stopping.`);
      stopAll(0);
    }, EXIT_AFTER * 1000);
  }
}

function banner(lines) {
  const width = Math.max(...lines.map((l) => l.length)) + 4;
  console.log(`\n  ${COLORS.bold("SARAS AI")} — Smart Assessment and Recognition Automation System`);
  console.log(`  ${"─".repeat(width)}`);
  for (const line of lines) console.log(`    ${line}`);
  console.log(`  ${"─".repeat(width)}`);
  console.log("    Press Ctrl+C to stop.\n");
}

// ----------------------------------------------------------------- commands
async function cmdDev({ api = true, web = true } = {}) {
  if (api) {
    requireBackend();
    requireVenv();
    if (await portInUse(API_PORT)) {
      die(`Port ${API_PORT} is already in use — is the API already running in another terminal? Stop it, or set SARAS_API_PORT.`);
    }
  }
  if (web) {
    ensureWebDeps();
    warnIfWebIncomplete();
    if (await portInUse(WEB_PORT)) {
      die(`Port ${WEB_PORT} is already in use — is the web dev server already running? Stop it, or set SARAS_WEB_PORT.`);
    }
    if (API_PORT !== 8000) say("Note: the web dev server proxies /api to port 8000 (frontend/vite.config.ts).", "error");
  }

  if (api) startApi({ reload: true });
  if (web) startWeb();

  const lines = [];
  if (api) {
    if (!(await waitForHttp(healthUrl(), 120_000))) {
      if (!stopping) {
        say(`The API did not answer ${healthUrl()} within 120 s.`, "error");
        stopAll(1);
      }
      return;
    }
  }
  if (web && !(await waitForHttp(`http://localhost:${WEB_PORT}/`, 60_000))) {
    if (!stopping) {
      say(`The web dev server did not start on port ${WEB_PORT} within 60 s.`, "error");
      stopAll(1);
    }
    return;
  }
  if (web) lines.push(`Web app   http://localhost:${WEB_PORT}`);
  if (api) lines.push(`API       http://127.0.0.1:${API_PORT}   (docs: http://127.0.0.1:${API_PORT}/api/docs)`);
  banner(lines);
  scheduleSmokeExit();
}

async function cmdStart(args) {
  requireBackend();
  requireVenv();
  const index = path.join(FRONTEND, "dist", "index.html");
  if (args.includes("--build") || !existsSync(index)) {
    ensureWebDeps();
    say("Building the web app (frontend/dist)…");
    if (runNpm(["run", "build"], FRONTEND) !== 0) die("The web app build failed — see the output above.");
  }
  if (await portInUse(API_PORT)) die(`Port ${API_PORT} is already in use. Stop the other server, or set SARAS_API_PORT.`);
  startApi({ reload: false });
  if (!(await waitForHttp(healthUrl(), 120_000))) {
    if (!stopping) {
      say(`The server did not answer ${healthUrl()} within 120 s.`, "error");
      stopAll(1);
    }
    return;
  }
  const shown = API_HOST === "0.0.0.0" ? "127.0.0.1" : API_HOST;
  banner([`Open      http://${shown}:${API_PORT}`, `API docs  http://${shown}:${API_PORT}/api/docs`]);
  scheduleSmokeExit();
}

function cmdBuild() {
  ensureWebDeps();
  process.exit(runNpm(["run", "build"], FRONTEND));
}

function cmdInstallWeb() {
  if (process.env.SARAS_SKIP_WEB_INSTALL) return;
  if (!existsSync(path.join(FRONTEND, "package.json"))) return;
  say("Installing the web app's dependencies in frontend/ …");
  process.exit(runNpm(["install", "--no-audit", "--no-fund"], FRONTEND));
}

function cmdPython(moduleArgs, missingMessage) {
  const python = requireVenv();
  if (missingMessage) die(missingMessage);
  const result = spawnSync(python, moduleArgs, { cwd: BACKEND, stdio: "inherit" });
  process.exit(result.status ?? 1);
}

function findSystemPython() {
  const candidates = IS_WIN
    ? [["py", ["-3.12"]], ["py", ["-3"]], ["python", []]]
    : [["python3.12", []], ["python3", []], ["python", []]];
  for (const [command, prefix] of candidates) {
    const probe = spawnSync(command, [...prefix, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], {
      encoding: "utf8",
      windowsHide: true,
    });
    if (probe.status === 0) {
      const [major, minor] = probe.stdout.trim().split(".").map(Number);
      if (major === 3 && minor >= 10) return [command, prefix];
    }
  }
  return null;
}

function cmdSetup(args) {
  if (!existsSync(VENV_PY)) {
    const python = findSystemPython();
    if (!python) die("Python 3.10 or newer was not found. Install Python 3.12 from python.org, then run: npm run setup");
    say(`Creating ${rel(path.join(BACKEND, ".venv"))} with ${[python[0], ...python[1]].join(" ")} …`);
    const created = spawnSync(python[0], [...python[1], "-m", "venv", path.join(BACKEND, ".venv")], { stdio: "inherit" });
    if (created.status !== 0 || !existsSync(VENV_PY)) die("Creating the virtual environment failed.");
  } else {
    say(`Using the existing virtual environment ${rel(path.dirname(path.dirname(VENV_PY)))}`);
  }

  const requirements = path.join(BACKEND, args.includes("--dev") ? "requirements-dev.txt" : "requirements.txt");
  if (existsSync(requirements)) {
    say(`Installing Python packages from ${rel(requirements)} …`);
    const pip = spawnSync(VENV_PY, ["-m", "pip", "install", "-r", requirements], { cwd: BACKEND, stdio: "inherit" });
    if (pip.status !== 0) die("pip install failed — see the output above.");
  } else {
    say(`${rel(requirements)} not found; skipped the Python packages.`, "error");
  }

  if (!args.includes("--no-web")) {
    say("Installing the web app's dependencies …");
    if (runNpm(["install", "--no-audit", "--no-fund"], FRONTEND) !== 0) die("npm install in frontend/ failed.");
  }
  console.log("\nSetup complete. Next:\n  npm run dev     # API + web app with auto-reload\n  npm start       # build once and serve everything on http://127.0.0.1:8000\n");
}

function printHelp() {
  console.log(`SARAS AI launcher — run from the repository root:

  npm install        install the web app's dependencies (frontend/)
  npm run setup      create backend/.venv, install Python + web dependencies (--dev adds test tools)
  npm run dev        API on :${API_PORT} (auto-reload) + web app on :${WEB_PORT}
  npm run dev:api    API only
  npm run dev:web    web app only (expects the API on :8000)
  npm run build      build the web app into frontend/dist
  npm start          build if needed, then serve everything from FastAPI on :${API_PORT} (add -- --build to rebuild)
  npm test           backend tests (e.g. npm test -- -k auth)
  npm run seed       load the demo teacher, exam and sample sheets

Overrides: SARAS_API_HOST, SARAS_API_PORT, SARAS_WEB_PORT.`);
}

// --------------------------------------------------------------------- main
const [command = "help", ...rest] = process.argv.slice(2);
switch (command) {
  case "dev":
    await cmdDev();
    break;
  case "api":
    await cmdDev({ web: false });
    break;
  case "web":
    await cmdDev({ api: false });
    break;
  case "start":
    await cmdStart(rest);
    break;
  case "build":
    cmdBuild();
    break;
  case "install-web":
    cmdInstallWeb();
    break;
  case "test":
    cmdPython(["-m", "pytest", ...(rest.length ? rest : ["-q"])]);
    break;
  case "seed":
    cmdPython(
      ["-m", "scripts.seed_demo", ...rest],
      existsSync(path.join(BACKEND, "scripts", "seed_demo.py")) ? null : "backend/scripts/seed_demo.py does not exist yet, so there is no demo data to load.",
    );
    break;
  case "setup":
    cmdSetup(rest);
    break;
  case "help":
  case "--help":
  case "-h":
    printHelp();
    break;
  default:
    die(`Unknown command "${command}". Run: node scripts/saras.mjs help`);
}
