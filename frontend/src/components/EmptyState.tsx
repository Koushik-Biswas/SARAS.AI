import type { ReactNode } from "react";
import { cx } from "./cx";

export type IllustrationKind = "exams" | "upload" | "review" | "chart" | "results" | "generic";

/** Inline SVG illustrations tinted entirely by theme tokens (see .ill-* in components.css). */
export function Illustration({ kind, size = 140 }: { kind: IllustrationKind; size?: number }) {
  const common = { width: size, height: Math.round(size * 0.78), viewBox: "0 0 180 140", "aria-hidden": true as const, focusable: false as const };
  switch (kind) {
    case "exams":
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="52" y="30" width="70" height="88" rx="8" transform="rotate(-8 87 74)" />
          <rect className="ill-paper" x="60" y="24" width="70" height="88" rx="8" />
          <rect className="ill-accent" x="70" y="38" width="30" height="6" rx="3" />
          <rect className="ill-line-fill" x="70" y="52" width="48" height="4" rx="2" />
          <rect className="ill-line-fill" x="70" y="62" width="42" height="4" rx="2" />
          <rect className="ill-line-fill" x="70" y="72" width="46" height="4" rx="2" />
          <rect className="ill-line-fill" x="70" y="82" width="30" height="4" rx="2" />
          <circle className="ill-accent" cx="128" cy="100" r="16" />
          <path className="ill-check" d="M120 100l6 6 10-12" />
        </svg>
      );
    case "upload":
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="46" y="58" width="88" height="56" rx="10" />
          <path className="ill-stroke" d="M46 86h26l6 10h24l6-10h26" />
          <circle className="ill-accent" cx="90" cy="42" r="20" />
          <path className="ill-check" d="M90 52V33m-8 8l8-8 8 8" />
        </svg>
      );
    case "review":
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="50" y="26" width="80" height="92" rx="10" />
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <circle className={i < 2 ? "ill-accent" : "ill-line-fill"} cx="66" cy={50 + i * 22} r="7" />
              {i < 2 ? <path className="ill-check ill-check--sm" d={`M62.5 ${50 + i * 22}l2.5 2.5 4.5-5`} /> : null}
              <rect className="ill-line-fill" x="80" y={47 + i * 22} width={i === 1 ? 30 : 38} height="5" rx="2.5" />
            </g>
          ))}
        </svg>
      );
    case "chart":
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="40" y="30" width="100" height="84" rx="10" />
          <path className="ill-stroke" d="M54 100h72" />
          <rect className="ill-line-fill" x="60" y="72" width="10" height="28" rx="3" />
          <rect className="ill-accent" x="76" y="56" width="10" height="44" rx="3" />
          <rect className="ill-line-fill" x="92" y="66" width="10" height="34" rx="3" />
          <rect className="ill-accent" x="108" y="46" width="10" height="54" rx="3" />
        </svg>
      );
    case "results":
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="36" y="34" width="108" height="76" rx="10" />
          <rect className="ill-accent" x="36" y="34" width="108" height="16" rx="8" />
          {[0, 1, 2].map((i) => (
            <g key={i}>
              <rect className="ill-line-fill" x="48" y={60 + i * 15} width="36" height="5" rx="2.5" />
              <rect className="ill-line-fill" x="96" y={60 + i * 15} width="14" height="5" rx="2.5" />
              <rect className="ill-line-fill" x="118" y={60 + i * 15} width="14" height="5" rx="2.5" />
            </g>
          ))}
        </svg>
      );
    default:
      return (
        <svg {...common} className="ill">
          <ellipse className="ill-bg" cx="90" cy="76" rx="74" ry="56" />
          <rect className="ill-paper" x="56" y="32" width="68" height="84" rx="10" />
          <circle className="ill-accent" cx="90" cy="66" r="14" />
          <rect className="ill-line-fill" x="70" y="92" width="40" height="5" rx="2.5" />
        </svg>
      );
  }
}

interface EmptyStateProps {
  title: ReactNode;
  description?: ReactNode;
  illustration?: IllustrationKind;
  icon?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
  className?: string;
}

export function EmptyState({ title, description, illustration = "generic", icon, action, compact, className }: EmptyStateProps) {
  return (
    <div className={cx("empty", compact && "empty--compact", className)}>
      <div className="empty__art">{icon ?? <Illustration kind={illustration} size={compact ? 104 : 148} />}</div>
      <h3 className="empty__title">{title}</h3>
      {description ? <p className="empty__text">{description}</p> : null}
      {action ? <div className="empty__action">{action}</div> : null}
    </div>
  );
}
