// Hooks shared by the review screen and the region editor.
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { useBlocker } from "react-router-dom";
import * as api from "../api/endpoints";
import type { CapabilitiesOut } from "../api/types";

// ---------------------------------------------------------------- keyboard
/** True when the key event comes from a text-entry control (shortcuts must not fire there). */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "INPUT") {
    const type = (target as HTMLInputElement).type;
    return !["checkbox", "radio", "button", "submit", "reset", "range", "color", "file"].includes(type);
  }
  return false;
}

/** Composite widgets that own their arrow keys (segmented controls, menus, splitters...). */
export function inArrowWidget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    !!target.closest('[role="radiogroup"], [role="tablist"], [role="menu"], [role="listbox"], [role="separator"], [role="slider"], [data-own-arrows]')
  );
}

/** A modal dialog is open: page shortcuts must stay quiet behind it. */
export function modalOpen(): boolean {
  return !!document.querySelector('[role="dialog"][aria-modal="true"]');
}

/** Window-level keydown listener that always sees the latest handler. */
export function useHotkeys(handler: (e: KeyboardEvent) => void, enabled = true): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.isComposing) return;
      ref.current(e);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

// ---------------------------------------------------------------- unsaved changes
/** Ask the user, then call `resume` (discard and continue) or `cancel` (stay). */
export type UnsavedPrompt = (resume: () => void, cancel: () => void) => void;

// The screen currently holding unsaved changes (at most one is mounted at a time).
let activePrompt: UnsavedPrompt | null = null;

/**
 * Run `action` now or, while a screen holds unsaved changes, only once the user agrees to
 * discard them. For actions that are not plain navigations, e.g. signing out.
 */
export function runGuarded(action: () => void): void {
  if (activePrompt) activePrompt(action, () => undefined);
  else action();
}

/**
 * While `active`: warn before the tab is closed or reloaded, and hold every in-app navigation
 * to another page (links, tabs, breadcrumbs, navigate() calls such as the sidebar's
 * "New exam", and the browser's Back/Forward buttons) plus runGuarded() actions until the
 * user decides through `onBlock`. Navigations the screen starts itself after its own
 * confirmation go through the returned `allow()`.
 */
export function useUnsavedGuard(active: boolean, onBlock: UnsavedPrompt): { allow: (navigate: () => void) => void } {
  const cb = useRef(onBlock);
  cb.current = onBlock;
  // Read through refs: the router keeps the blocker function from an earlier render.
  const activeRef = useRef(active);
  activeRef.current = active;
  const bypass = useRef(false);

  const blocker = useBlocker(
    useCallback(
      ({ currentLocation, nextLocation }: { currentLocation: { pathname: string }; nextLocation: { pathname: string } }) =>
        activeRef.current && !bypass.current && currentLocation.pathname !== nextLocation.pathname,
      [],
    ),
  );
  useEffect(() => {
    if (blocker.state !== "blocked") return;
    if (!activeRef.current) {
      blocker.proceed(); // the changes were saved meanwhile
      return;
    }
    cb.current(
      () => blocker.proceed(),
      () => blocker.reset(),
    );
  }, [blocker]);

  const allow = useCallback((navigate: () => void) => {
    bypass.current = true; // the router consults the blocker synchronously inside navigate()
    try {
      navigate();
    } finally {
      bypass.current = false;
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      // Legacy browsers need a returnValue to show the prompt.
      e.returnValue = "";
    };
    const prompt: UnsavedPrompt = (resume, cancel) => cb.current(resume, cancel);
    activePrompt = prompt;
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      if (activePrompt === prompt) activePrompt = null;
    };
  }, [active]);

  return { allow };
}

// ---------------------------------------------------------------- visibility
/** Becomes true (and stays true) once the element scrolls within `rootMargin` of the viewport. */
export function useInView<T extends Element>(ref: RefObject<T>, enabled = true, rootMargin = "600px"): boolean {
  const [seen, setSeen] = useState(!enabled);
  useEffect(() => {
    if (seen || !enabled) return;
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((en) => en.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, enabled, rootMargin, seen]);
  return seen || !enabled;
}

// ---------------------------------------------------------------- page images
// Small LRU of in-flight/fetched blobs so switching answers or pages never refetches a
// page already seen. Each consumer makes (and revokes) its own object URL.
const LIMIT = { full: 14, thumb: 80 } as const;
const blobCache = { full: new Map<string, Promise<Blob>>(), thumb: new Map<string, Promise<Blob>>() };

function loadPageBlob(submissionId: string, pageNo: number, thumb: boolean): Promise<Blob> {
  const kind = thumb ? "thumb" : "full";
  const cache = blobCache[kind];
  const key = `${submissionId}:${pageNo}`;
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit); // bump to most-recent
    return hit;
  }
  const p = api.getPageImage(submissionId, pageNo, thumb);
  cache.set(key, p);
  p.catch(() => {
    if (cache.get(key) === p) cache.delete(key);
  });
  while (cache.size > LIMIT[kind]) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
  return p;
}

export interface PageImageState {
  url: string | null;
  error: unknown;
  loading: boolean;
  retry: () => void;
}

/** Fetch a page image (bearer auth) as a blob URL. Nothing is fetched while `enabled` is false. */
export function usePageImage(submissionId: string, pageNo: number, thumb = false, enabled = true): PageImageState {
  const [state, setState] = useState<{ url: string | null; error: unknown; loading: boolean }>({
    url: null,
    error: null,
    loading: enabled,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let url: string | null = null;
    setState({ url: null, error: null, loading: true });
    loadPageBlob(submissionId, pageNo, thumb).then(
      (blob) => {
        if (!alive) return;
        url = URL.createObjectURL(blob);
        setState({ url, error: null, loading: false });
      },
      (err: unknown) => {
        if (alive) setState({ url: null, error: err, loading: false });
      },
    );
    return () => {
      alive = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [submissionId, pageNo, thumb, enabled, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { ...state, retry };
}

// ---------------------------------------------------------------- capabilities
let capsPromise: Promise<CapabilitiesOut> | null = null;

/** Server thresholds (auto-accept, contradiction) for meters; fetched once per session. */
export function useCapabilities(): CapabilitiesOut | null {
  const [caps, setCaps] = useState<CapabilitiesOut | null>(null);
  useEffect(() => {
    let alive = true;
    if (!capsPromise) {
      capsPromise = api.getCapabilities();
      capsPromise.catch(() => {
        capsPromise = null;
      });
    }
    capsPromise.then(
      (c) => {
        if (alive) setCaps(c);
      },
      () => undefined,
    );
    return () => {
      alive = false;
    };
  }, []);
  return caps;
}
