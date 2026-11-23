import { useDelayedFlag } from "../hooks/useDelayedFlag";
import { cx } from "./cx";

interface SpinnerProps {
  size?: number;
  label?: string;
  className?: string;
}

/** Immediate spinner (use inside buttons). For page loading prefer <DelayedSpinner>. */
export function Spinner({ size = 16, label = "Loading", className }: SpinnerProps) {
  return (
    <span className={cx("spinner", className)} role="status" style={{ width: size, height: size }}>
      <span className="sr-only">{label}</span>
    </span>
  );
}

/** Spinner that only appears after `delayMs` (default 300 ms) of continuous loading. */
export function DelayedSpinner({
  loading,
  delayMs = 300,
  label,
  size = 18,
}: {
  loading: boolean;
  delayMs?: number;
  label?: string;
  size?: number;
}) {
  const show = useDelayedFlag(loading, delayMs);
  if (!show) return null;
  return (
    <span className="delayed-spinner">
      <Spinner size={size} label={label} />
      {label ? <span className="muted">{label}</span> : null}
    </span>
  );
}
