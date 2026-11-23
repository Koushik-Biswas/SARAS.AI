import type { ReactNode } from "react";
import { CircleCheck, CircleX, Info, RefreshCw, TriangleExclamationPoint } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import { Button } from "./Button";
import { cx } from "./cx";

type BannerTone = "danger" | "warning" | "info" | "success";
const ICONS = { danger: CircleX, warning: TriangleExclamationPoint, info: Info, success: CircleCheck };

interface BannerProps {
  tone?: BannerTone;
  title?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}

/** Inline message strip (icon + text; tone never carries meaning alone). */
export function Banner({ tone = "info", title, children, action, className }: BannerProps) {
  const Icon = ICONS[tone];
  return (
    <div className={cx("banner", `banner--${tone}`, className)} role={tone === "danger" ? "alert" : "status"}>
      <Icon className="banner__icon" size={18} aria-hidden="true" />
      <div className="banner__content">
        {title ? <p className="banner__title">{title}</p> : null}
        {children ? <div className="banner__text">{children}</div> : null}
      </div>
      {action ? <div className="banner__action">{action}</div> : null}
    </div>
  );
}

interface ErrorBannerProps {
  error: unknown;
  onRetry?: () => void;
  title?: string;
  retrying?: boolean;
  className?: string;
}

/** Error strip with a Retry action. Accepts ApiError or any thrown value. */
export function ErrorBanner({ error, onRetry, title, retrying, className }: ErrorBannerProps) {
  if (!error) return null;
  const isNetwork = error instanceof ApiError && error.isNetwork;
  const heading = title ?? (isNetwork ? "Cannot reach the server" : "Something went wrong");
  return (
    <Banner
      tone="danger"
      title={heading}
      className={className}
      action={
        onRetry ? (
          <Button size="sm" variant="secondary" onClick={onRetry} loading={retrying} iconLeft={<RefreshCw size={14} aria-hidden="true" />}>
            Retry
          </Button>
        ) : undefined
      }
    >
      {errorMessage(error)}
    </Banner>
  );
}
