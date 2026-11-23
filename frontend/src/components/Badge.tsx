import type { ReactNode } from "react";
import {
  Ban,
  CircleCheck,
  CircleCheckBig,
  CircleDashed,
  CircleExclamationPoint,
  CircleMinus,
  CircleX,
  Clock,
  Eye,
  Flag,
  LoaderCircle,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import type { AnswerStatus, KeyPointStatus, SubmissionStatus } from "../api/types";
import { cx } from "./cx";

export type Tone = "neutral" | "accent" | "success" | "warning" | "danger" | "info";

interface BadgeProps {
  tone?: Tone;
  icon?: ReactNode;
  children: ReactNode;
  size?: "sm" | "md";
  className?: string;
  title?: string;
}

export function Badge({ tone = "neutral", icon, children, size = "md", className, title }: BadgeProps) {
  return (
    <span className={cx("badge", `badge--${tone}`, `badge--${size}`, className)} title={title}>
      {icon}
      <span>{children}</span>
    </span>
  );
}

/** Small numeric count bubble (tabs, nav). */
export function CountBadge({ count, tone = "neutral", label }: { count: number; tone?: Tone; label?: string }) {
  if (!count) return null;
  return (
    <span className={cx("count-badge", `count-badge--${tone}`)} aria-label={label ? `${count} ${label}` : undefined}>
      {count > 99 ? "99+" : count}
    </span>
  );
}

interface StatusSpec {
  tone: Tone;
  icon: LucideIcon;
  label: string;
  spin?: boolean;
}

const SUBMISSION: Record<SubmissionStatus, StatusSpec> = {
  uploaded: { tone: "neutral", icon: Clock, label: "Queued" },
  processing: { tone: "info", icon: LoaderCircle, label: "Processing", spin: true },
  processed: { tone: "accent", icon: Eye, label: "Ready for review" },
  in_review: { tone: "warning", icon: Eye, label: "In review" },
  approved: { tone: "success", icon: ShieldCheck, label: "Approved" },
  failed: { tone: "danger", icon: CircleX, label: "Failed" },
};

const ANSWER: Record<AnswerStatus, StatusSpec> = {
  needs_review: { tone: "warning", icon: Flag, label: "Needs review" },
  auto_accepted: { tone: "info", icon: CircleCheck, label: "Auto-accepted" },
  reviewed: { tone: "success", icon: CircleCheckBig, label: "Reviewed" },
  blank: { tone: "neutral", icon: CircleDashed, label: "Blank" },
  failed: { tone: "danger", icon: CircleExclamationPoint, label: "Failed" },
};

const KEYPOINT: Record<KeyPointStatus, StatusSpec> = {
  matched: { tone: "success", icon: CircleCheck, label: "Matched" },
  partial: { tone: "warning", icon: CircleMinus, label: "Partial" },
  missed: { tone: "neutral", icon: CircleX, label: "Missed" },
  contradicted: { tone: "danger", icon: Ban, label: "Contradicted" },
};

const FALLBACK: StatusSpec = { tone: "neutral", icon: CircleDashed, label: "Unknown" };

export type StatusKind = "submission" | "answer" | "keypoint";

export function statusSpec(kind: StatusKind, status: string): StatusSpec {
  const table = (kind === "submission" ? SUBMISSION : kind === "answer" ? ANSWER : KEYPOINT) as Record<string, StatusSpec>;
  return table[status] ?? { ...FALLBACK, label: status.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()) };
}

export function statusLabel(kind: StatusKind, status: string): string {
  return statusSpec(kind, status).label;
}

/** Status shown as icon + text + tone (never colour alone). */
export function StatusPill({
  kind,
  status,
  size = "md",
  label,
}: {
  kind: StatusKind;
  status: string;
  size?: "sm" | "md";
  /** Override the text label. */
  label?: string;
}) {
  const spec = statusSpec(kind, status);
  const Icon = spec.icon;
  return (
    <Badge tone={spec.tone} size={size} icon={<Icon size={size === "sm" ? 12 : 14} aria-hidden="true" className={spec.spin ? "spin" : undefined} />}>
      {label ?? spec.label}
    </Badge>
  );
}
