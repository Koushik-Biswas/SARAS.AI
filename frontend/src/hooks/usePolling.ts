import { useEffect, useRef } from "react";

/**
 * Calls `tick` every `intervalMs` while `enabled`, pausing while the tab is hidden.
 * The next tick is scheduled only after the previous one settles (no overlap).
 */
export function usePolling(tick: () => Promise<unknown> | void, intervalMs: number, enabled: boolean): void {
  const tickRef = useRef(tick);
  tickRef.current = tick;

  useEffect(() => {
    if (!enabled) return;
    let timer: number | undefined;
    let stopped = false;
    const schedule = () => {
      if (stopped) return;
      timer = window.setTimeout(async () => {
        if (document.visibilityState === "visible") {
          try {
            await tickRef.current();
          } catch {
            /* surfaced by the caller's own state */
          }
        }
        schedule();
      }, intervalMs);
    };
    schedule();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [enabled, intervalMs]);
}
