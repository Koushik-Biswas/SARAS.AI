// Thin fetch wrapper for the SARAS AI API (/api/v1).
// - Access token lives in memory only; refresh token in localStorage (shared by every tab).
// - On 401 the client refreshes once and retries the request. Refreshes are single-flight per
//   tab and serialised across tabs (Web Locks), because the refresh token rotates on every use.
// - A rejected refresh only drops the stored token if it is still the one that was rejected:
//   another tab may have rotated a fresh one in meanwhile (then that one is tried instead).
//   Outages (network, 429, 5xx) never sign anyone out.
// - Error bodies {"error": {code, message, details}} become ApiError instances.

import type { FieldErrorDetail, TokenOut, UploadProgress } from "./types";

// Same origin by default (FastAPI serves this SPA). For a split deployment (e.g. the SPA
// on Vercel, the API elsewhere) build with VITE_API_ORIGIN=https://api.example.com and
// add the SPA's origin to the backend's SARAS_CORS_ORIGINS.
const API_ORIGIN = ((import.meta.env.VITE_API_ORIGIN as string | undefined) ?? "").trim().replace(/\/+$/, "");
export const API_BASE = `${API_ORIGIN}/api/v1`;
const REFRESH_KEY = "saras.refresh_token";

// ---------------------------------------------------------------- errors
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /** Field-level errors from 400 validation details, keyed by dotted field path. */
  get fieldErrors(): Record<string, string> {
    const out: Record<string, string> = {};
    const add = (d: unknown) => {
      if (d && typeof d === "object" && "field" in d && "message" in d) {
        const fe = d as FieldErrorDetail;
        const key = String(fe.field ?? "");
        if (!(key in out)) out[key] = String(fe.message);
      }
    };
    if (Array.isArray(this.details)) this.details.forEach(add);
    else add(this.details);
    return out;
  }

  /** Seconds to wait before retrying (429), from details.retry_after_seconds. */
  get retryAfterSeconds(): number | null {
    const d = this.details as { retry_after_seconds?: unknown } | null | undefined;
    const v = d && typeof d === "object" ? d.retry_after_seconds : undefined;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  get isNetwork(): boolean {
    return this.status === 0;
  }
}

/** Human-readable message for any thrown value. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong";
}

async function parseError(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let message = res.statusText || "Request failed";
  let details: unknown = undefined;
  try {
    const text = await res.text();
    if (text) {
      const body = JSON.parse(text) as { error?: { code?: string; message?: string; details?: unknown } };
      if (body && body.error) {
        code = body.error.code || code;
        message = body.error.message || message;
        details = body.error.details;
      }
    }
  } catch {
    /* non-JSON body: keep defaults */
  }
  if (res.status === 429 && details === undefined) {
    const ra = Number(res.headers.get("Retry-After"));
    if (Number.isFinite(ra) && ra > 0) details = { retry_after_seconds: ra };
  }
  return new ApiError(res.status, code, message, details);
}

function networkError(): ApiError {
  return new ApiError(0, "network_error", "Cannot reach the server. Check your connection and try again.");
}

// ---------------------------------------------------------------- session store
type SessionListener = (user: TokenOut["user"] | null) => void;

type RefreshResult = "ok" | "rejected" | "unavailable";

let accessToken: string | null = null;
let refreshInFlight: Promise<RefreshResult> | null = null;
const listeners = new Set<SessionListener>();

function readRefresh(): string | null {
  try {
    return localStorage.getItem(REFRESH_KEY);
  } catch {
    return null;
  }
}

function writeRefresh(token: string | null): void {
  try {
    if (token) localStorage.setItem(REFRESH_KEY, token);
    else localStorage.removeItem(REFRESH_KEY);
  } catch {
    /* storage unavailable (private mode): session will not survive reload */
  }
}

export const session = {
  get accessToken(): string | null {
    return accessToken;
  },
  get refreshToken(): string | null {
    return readRefresh();
  },
  hasRefreshToken(): boolean {
    return !!readRefresh();
  },
  set(tokens: TokenOut): void {
    accessToken = tokens.access_token;
    writeRefresh(tokens.refresh_token);
    listeners.forEach((l) => l(tokens.user));
  },
  clear(): void {
    const had = accessToken !== null || readRefresh() !== null;
    accessToken = null;
    writeRefresh(null);
    if (had) listeners.forEach((l) => l(null));
  },
  subscribe(listener: SessionListener): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

/**
 * The server refused `rejected` (or there was no token at all): forget this tab's access token,
 * and the stored refresh token too, but only if it is still the rejected one (compare-and-clear).
 */
function expire(rejected: string | null): void {
  const stored = readRefresh();
  const had = accessToken !== null || (rejected !== null && stored === rejected);
  accessToken = null;
  if (rejected !== null && stored === rejected) writeRefresh(null);
  if (had) listeners.forEach((l) => l(null));
}

// Another tab signed out (or its refresh was refused): this tab's session is over too.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => {
    if ((e.key === REFRESH_KEY || e.key === null) && readRefresh() === null && accessToken !== null) {
      accessToken = null;
      listeners.forEach((l) => l(null));
    }
  });
}

/** Serialise refreshes across tabs: they all rotate the same stored token. */
function withRefreshLock<T>(fn: () => Promise<T>): Promise<T> {
  const locks = typeof navigator !== "undefined" ? (navigator as Navigator & { locks?: LockManager }).locks : undefined;
  if (locks && typeof locks.request === "function") return locks.request("saras-auth-refresh", () => fn()) as Promise<T>;
  return fn();
}

async function refreshOnce(): Promise<RefreshResult> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const token = readRefresh(); // read inside the lock: another tab may just have rotated it
    if (!token) {
      expire(null);
      return "rejected";
    }
    let res: Response;
    try {
      res = await fetch(`${API_BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ refresh_token: token }),
      });
      if (res.ok) {
        session.set((await res.json()) as TokenOut);
        return "ok";
      }
    } catch {
      return "unavailable";
    }
    // Only drop the session when the server rejected the token, not on outages.
    if (res.status !== 401 && res.status !== 400 && res.status !== 403) return "unavailable";
    const now = readRefresh();
    if (now && now !== token) continue; // rotated by another tab while we asked: try the new one
    expire(token);
    return "rejected";
  }
  return "unavailable";
}

function refreshDetailed(): Promise<RefreshResult> {
  if (refreshInFlight) return refreshInFlight;
  if (!readRefresh()) {
    expire(null);
    return Promise.resolve("rejected");
  }
  refreshInFlight = withRefreshLock(refreshOnce)
    .catch((): RefreshResult => "unavailable")
    .finally(() => {
      refreshInFlight = null;
    });
  return refreshInFlight;
}

/** Exchange the stored refresh token for a new pair. Single-flight; resolves false on failure. */
export function refreshSession(): Promise<boolean> {
  return refreshDetailed().then((r) => r === "ok");
}

// ---------------------------------------------------------------- core request
export type Query = Record<string, string | number | boolean | null | undefined>;

export interface RequestOptions {
  method?: string;
  query?: Query;
  /** JSON-serialisable body, or FormData / Blob sent as-is. */
  body?: unknown;
  headers?: Record<string, string>;
  /** Attach the bearer token and refresh on 401 (default true). */
  auth?: boolean;
  signal?: AbortSignal;
}

export function buildUrl(path: string, query?: Query): string {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  if (!query) return url;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    qs.append(k, String(v));
  }
  const s = qs.toString();
  return s ? `${url}?${s}` : url;
}

async function rawFetch(path: string, opts: RequestOptions): Promise<Response> {
  const { method = "GET", query, body, headers = {}, auth = true, signal } = opts;
  const h: Record<string, string> = { Accept: "application/json", ...headers };
  let payload: BodyInit | undefined;
  if (body instanceof FormData || body instanceof Blob) payload = body;
  else if (body !== undefined) {
    h["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  const send = () => {
    const hh = { ...h };
    if (auth && accessToken) hh.Authorization = `Bearer ${accessToken}`;
    return fetch(buildUrl(path, query), { method, headers: hh, body: payload, signal });
  };

  let res: Response;
  try {
    if (auth && !accessToken && readRefresh()) await refreshSession();
    res = await send();
    if (res.status === 401 && auth) {
      // "rejected" has already signed this tab out; "unavailable" (outage) keeps the session and
      // the caller just sees this error.
      if ((await refreshDetailed()) !== "ok") return res;
      res = await send();
      if (res.status === 401) session.clear(); // a brand-new token refused: the account is gone
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw networkError();
  }
  return res;
}

/** Perform a request and return parsed JSON (undefined for 204). Throws ApiError. */
export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await rawFetch(path, opts);
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const http = {
  get: <T>(path: string, query?: Query, signal?: AbortSignal) => request<T>(path, { query, signal }),
  post: <T>(path: string, body?: unknown, extra: Omit<RequestOptions, "method" | "body"> = {}) =>
    request<T>(path, { ...extra, method: "POST", body }),
  put: <T>(path: string, body?: unknown) => request<T>(path, { method: "PUT", body }),
  patch: <T>(path: string, body?: unknown) => request<T>(path, { method: "PATCH", body }),
  del: <T = void>(path: string) => request<T>(path, { method: "DELETE" }),
};

// ---------------------------------------------------------------- blobs / downloads
/** Fetch a binary resource with bearer auth (e.g. page images). */
export async function fetchBlob(path: string, query?: Query, signal?: AbortSignal): Promise<Blob> {
  const res = await rawFetch(path, { query, signal, headers: { Accept: "*/*" } });
  if (!res.ok) throw await parseError(res);
  return res.blob();
}

function filenameFrom(res: Response, fallback: string): string {
  const cd = res.headers.get("Content-Disposition") || "";
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  if (star) return decodeURIComponent(star[1]);
  const plain = /filename="?([^";]+)"?/i.exec(cd);
  return plain ? plain[1] : fallback;
}

/** Download an authenticated file (CSV export) and trigger a browser save. */
export async function downloadFile(path: string, query: Query | undefined, fallbackName: string): Promise<string> {
  const res = await rawFetch(path, { query, headers: { Accept: "text/csv,*/*" } });
  if (!res.ok) throw await parseError(res);
  const blob = await res.blob();
  const name = filenameFrom(res, fallbackName);
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return name;
}

// ---------------------------------------------------------------- multipart upload with progress
export function newIdempotencyKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  // Fallback for non-secure contexts.
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export interface UploadOptions {
  idempotencyKey?: string;
  onProgress?: (p: UploadProgress) => void;
  signal?: AbortSignal;
}

function xhrOnce<T>(path: string, form: FormData, opts: UploadOptions, key: string): Promise<{ status: number; xhr: XMLHttpRequest; data?: T }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", buildUrl(path));
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("Idempotency-Key", key);
    if (accessToken) xhr.setRequestHeader("Authorization", `Bearer ${accessToken}`);
    xhr.upload.onprogress = (ev) => {
      if (opts.onProgress && ev.lengthComputable) {
        opts.onProgress({ loaded: ev.loaded, total: ev.total, fraction: ev.total ? ev.loaded / ev.total : 0 });
      }
    };
    xhr.onload = () => resolve({ status: xhr.status, xhr });
    xhr.onerror = () => reject(networkError());
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    if (opts.signal) {
      if (opts.signal.aborted) {
        reject(new DOMException("Upload cancelled", "AbortError"));
        return;
      }
      opts.signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(form);
  });
}

function xhrError(xhr: XMLHttpRequest): ApiError {
  try {
    const body = JSON.parse(xhr.responseText) as { error?: { code?: string; message?: string; details?: unknown } };
    if (body?.error) {
      return new ApiError(xhr.status, body.error.code || `http_${xhr.status}`, body.error.message || "Upload failed", body.error.details);
    }
  } catch {
    /* fall through */
  }
  return new ApiError(xhr.status, `http_${xhr.status}`, xhr.statusText || "Upload failed");
}

/**
 * POST multipart form data with upload progress (XMLHttpRequest) and an Idempotency-Key.
 * The same key is reused on the post-refresh retry so the server never creates duplicates.
 */
export async function uploadMultipart<T>(path: string, form: FormData, opts: UploadOptions = {}): Promise<T> {
  const key = opts.idempotencyKey ?? newIdempotencyKey();
  if (!accessToken && readRefresh()) await refreshSession();
  let r = await xhrOnce<T>(path, form, opts, key);
  if (r.status === 401) {
    if ((await refreshDetailed()) !== "ok") throw xhrError(r.xhr); // see rawFetch
    r = await xhrOnce<T>(path, form, opts, key);
    if (r.status === 401) session.clear();
  }
  if (r.status < 200 || r.status >= 300) throw xhrError(r.xhr);
  return (r.xhr.responseText ? JSON.parse(r.xhr.responseText) : undefined) as T;
}
