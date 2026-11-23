import { useCallback, useEffect, useRef, useState, type DependencyList } from "react";

export interface AsyncState<T> {
  data: T | undefined;
  error: unknown;
  /** True while the first load (or a non-silent reload) is in flight. */
  loading: boolean;
  /** True while any request is in flight, including silent refreshes. */
  fetching: boolean;
  /** Re-run the loader. `silent` keeps current data visible (used for polling). */
  reload: (silent?: boolean) => Promise<void>;
  setData: (updater: T | ((prev: T | undefined) => T)) => void;
}

/**
 * Minimal data loader: runs `fn` when `deps` change, ignores stale responses,
 * and exposes reload/setData for optimistic updates.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: DependencyList): AsyncState<T> {
  const [data, setDataState] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(undefined);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(true);
  const seq = useRef(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const run = useCallback(async (silent: boolean) => {
    const id = ++seq.current;
    setFetching(true);
    if (!silent) setLoading(true);
    try {
      const result = await fnRef.current();
      if (id !== seq.current) return;
      setDataState(result);
      setError(undefined);
    } catch (e) {
      if (id !== seq.current) return;
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e);
    } finally {
      if (id === seq.current) {
        setLoading(false);
        setFetching(false);
      }
    }
  }, []);

  useEffect(() => {
    void run(false);
    return () => {
      seq.current++; // invalidate in-flight responses on dependency change / unmount
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  const reload = useCallback((silent = false) => run(silent), [run]);
  const setData = useCallback((updater: T | ((prev: T | undefined) => T)) => {
    setDataState((prev) => (typeof updater === "function" ? (updater as (p: T | undefined) => T)(prev) : updater));
  }, []);

  return { data, error, loading, fetching, reload, setData };
}
