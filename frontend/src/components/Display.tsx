// Small presentational components: Card, Kbd, Avatar, Skeleton, ProgressBar, StatTile, ConfidenceMeter.
import type { CSSProperties, HTMLAttributes, ReactNode } from "react";
import { cx } from "./cx";

// ---------------------------------------------------------------- Card
interface CardProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
  /** Remove body padding (tables that run edge to edge). */
  flush?: boolean;
  interactive?: boolean;
  as?: "div" | "section" | "article";
}

export function Card({ title, subtitle, actions, footer, flush, interactive, as = "section", className, children, ...rest }: CardProps) {
  const Tag = as;
  return (
    <Tag className={cx("card", interactive && "card--interactive", className)} {...rest}>
      {title || actions ? (
        <header className="card__header">
          <div className="card__heading">
            {title ? <h3 className="card__title">{title}</h3> : null}
            {subtitle ? <p className="card__subtitle">{subtitle}</p> : null}
          </div>
          {actions ? <div className="card__actions">{actions}</div> : null}
        </header>
      ) : null}
      <div className={cx("card__body", flush && "card__body--flush")}>{children}</div>
      {footer ? <footer className="card__footer">{footer}</footer> : null}
    </Tag>
  );
}

// ---------------------------------------------------------------- Kbd
export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="kbd">{children}</kbd>;
}

// ---------------------------------------------------------------- Avatar
function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export function Avatar({ name, size = 32, className }: { name: string; size?: number; className?: string }) {
  return (
    <span
      className={cx("avatar", className)}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.38) }}
      aria-hidden="true"
    >
      {initials(name)}
    </span>
  );
}

// ---------------------------------------------------------------- Skeleton
interface SkeletonProps {
  width?: CSSProperties["width"];
  height?: CSSProperties["height"];
  radius?: CSSProperties["borderRadius"];
  className?: string;
  /** Render N text lines (last one shorter). */
  lines?: number;
}

export function Skeleton({ width = "100%", height = 14, radius, className, lines }: SkeletonProps) {
  if (lines && lines > 1) {
    return (
      <span className="skeleton-lines" aria-hidden="true">
        {Array.from({ length: lines }, (_, i) => (
          <span
            key={i}
            className={cx("skeleton", className)}
            style={{ width: i === lines - 1 ? "60%" : width, height, borderRadius: radius }}
          />
        ))}
      </span>
    );
  }
  return <span className={cx("skeleton", className)} style={{ width, height, borderRadius: radius }} aria-hidden="true" />;
}

/** Table-shaped skeleton for list screens. */
export function SkeletonTable({ rows = 5, cols = 5 }: { rows?: number; cols?: number }) {
  return (
    <div className="skeleton-table" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="skeleton-table__row" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: cols }, (_, c) => (
            <Skeleton key={c} width={c === 0 ? "70%" : `${40 + ((r + c) % 3) * 15}%`} />
          ))}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- ProgressBar
interface ProgressBarProps {
  /** 0..1; omit/null for indeterminate. */
  value?: number | null;
  label: string;
  showLabel?: boolean;
  tone?: "accent" | "success" | "warning" | "danger";
  size?: "sm" | "md";
  className?: string;
  /** Text shown at the right and announced (defaults to a percentage). */
  valueText?: string;
}

export function ProgressBar({ value, label, showLabel, tone = "accent", size = "md", className, valueText }: ProgressBarProps) {
  const indeterminate = value === undefined || value === null;
  const pct = indeterminate ? 0 : Math.max(0, Math.min(1, value)) * 100;
  const text = valueText ?? (indeterminate ? undefined : `${Math.round(pct)}%`);
  return (
    <div className={cx("progress-wrap", className)}>
      {showLabel ? (
        <div className="progress__meta">
          <span className="progress__label">{label}</span>
          {text ? <span className="progress__value num">{text}</span> : null}
        </div>
      ) : null}
      <div
        className={cx("progress", `progress--${size}`, `progress--${tone}`, indeterminate && "progress--indeterminate")}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={indeterminate ? undefined : Math.round(pct)}
        aria-valuetext={text}
      >
        <span className="progress__fill" style={indeterminate ? undefined : { width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- StatTile
interface StatTileProps {
  label: string;
  value: ReactNode;
  icon?: ReactNode;
  sub?: ReactNode;
  /** Optional change indicator, e.g. "+4 today". */
  delta?: { text: string; tone: "up" | "down" | "neutral" };
  tone?: "accent" | "success" | "warning" | "danger" | "info" | "neutral";
  loading?: boolean;
}

export function StatTile({ label, value, icon, sub, delta, tone = "accent", loading }: StatTileProps) {
  return (
    <div className={cx("stat", `stat--${tone}`)}>
      <div className="stat__top">
        <span className="stat__label">{label}</span>
        {icon ? (
          <span className="stat__icon" aria-hidden="true">
            {icon}
          </span>
        ) : null}
      </div>
      {loading ? (
        <Skeleton width="45%" height={30} radius={8} />
      ) : (
        <div className="stat__value num">{value}</div>
      )}
      {delta || sub ? (
        <div className="stat__foot">
          {delta ? <span className={cx("stat__delta", `stat__delta--${delta.tone}`)}>{delta.text}</span> : null}
          {sub ? <span className="stat__sub">{sub}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- ConfidenceMeter
export function confidenceLevel(value: number, tau = 0.75, tauLow = 0.5): "high" | "medium" | "low" {
  if (value >= tau) return "high";
  if (value >= tauLow) return "medium";
  return "low";
}

interface ConfidenceMeterProps {
  /** 0..1 */
  value: number | null | undefined;
  label?: string;
  tau?: number;
  tauLow?: number;
  compact?: boolean;
}

/** Bar + percentage + text label (High / Medium / Low): never colour alone. */
export function ConfidenceMeter({ value, label = "Confidence", tau, tauLow, compact }: ConfidenceMeterProps) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className="subtle">-</span>;
  }
  const v = Math.max(0, Math.min(1, value));
  const level = confidenceLevel(v, tau, tauLow);
  const word = level === "high" ? "High" : level === "medium" ? "Medium" : "Low";
  const pct = Math.round(v * 100);
  return (
    <span
      className={cx("meter", `meter--${level}`, compact && "meter--compact")}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-valuetext={`${pct}% (${word})`}
    >
      <span className="meter__track" aria-hidden="true">
        <span className="meter__fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="meter__pct num">{pct}%</span>
      {!compact ? <span className="meter__word">{word}</span> : null}
    </span>
  );
}
