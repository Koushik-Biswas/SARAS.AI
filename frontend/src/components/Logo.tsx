import { useId } from "react";
import { cx } from "./cx";

/** Rounded-square "S" monogram with a scan line + check motif, filled with the accent gradient tokens. */
export function LogoMark({ size = 32, className }: { size?: number; className?: string }) {
  const gid = useId().replace(/:/g, "");
  return (
    <svg className={cx("logo-mark", className)} width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id={`lg-${gid}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" style={{ stopColor: "var(--logo-1)" }} />
          <stop offset="1" style={{ stopColor: "var(--logo-2)" }} />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8.5" fill={`url(#lg-${gid})`} />
      <path
        className="logo-mark__s"
        d="M20.6 10.4c-.9-1.2-2.6-2-4.6-2-2.8 0-4.6 1.4-4.6 3.4 0 4.6 9.4 2.6 9.4 7.6 0 2.1-2 3.6-4.9 3.6-2.2 0-4-.9-5-2.3"
      />
      <path className="logo-mark__scan" d="M6.5 16h3M22.5 16h3" />
      <circle className="logo-mark__dot" cx="24.6" cy="24.6" r="3.4" />
      <path className="logo-mark__check" d="M23.1 24.6l1 1 1.9-2.1" />
    </svg>
  );
}

export function Logo({ collapsed, size = 32, tagline }: { collapsed?: boolean; size?: number; tagline?: boolean }) {
  return (
    <span className={cx("logo", collapsed && "logo--collapsed")}>
      <LogoMark size={size} />
      {!collapsed ? (
        <span className="logo__text">
          <span className="logo__word">
            SARAS <span className="logo__ai">AI</span>
          </span>
          {tagline ? <span className="logo__tagline">Smart Assessment and Recognition Automation System</span> : null}
        </span>
      ) : null}
    </span>
  );
}
