import { useEffect, useState } from "react";

/**
 * Returns true only after `flag` has stayed true for `delayMs` (default 300 ms).
 * Used so spinners never flash for fast responses (PRD 17 UX rules).
 */
export function useDelayedFlag(flag: boolean, delayMs = 300): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!flag) {
      setShown(false);
      return;
    }
    const t = window.setTimeout(() => setShown(true), delayMs);
    return () => window.clearTimeout(t);
  }, [flag, delayMs]);
  return flag && shown;
}
