import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Link, type LinkProps } from "react-router-dom";
import { cx } from "./cx";
import { Spinner } from "./Spinner";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "subtle";
export type ButtonSize = "sm" | "md" | "lg";

interface StyleProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  block?: boolean;
}

export function buttonClass({ variant = "secondary", size = "md", block }: StyleProps, extra?: string) {
  return cx("btn", `btn--${variant}`, `btn--${size}`, block && "btn--block", extra);
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, StyleProps {
  loading?: boolean;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, block, loading, iconLeft, iconRight, className, children, disabled, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={buttonClass({ variant, size, block }, cx(loading && "is-loading", className))}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size={size === "sm" ? 12 : 14} label="Working" /> : iconLeft}
      {children !== undefined && children !== null && children !== false ? (
        <span className="btn__label">{children}</span>
      ) : null}
      {iconRight}
    </button>
  );
});

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  /** Accessible name (also used as the native tooltip). */
  label: string;
  icon: ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, variant = "ghost", size = "md", loading, className, disabled, type = "button", title, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      aria-label={label}
      title={title ?? label}
      className={buttonClass({ variant, size }, cx("btn--icon", className))}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner size={14} label="Working" /> : icon}
    </button>
  );
});

export interface ButtonLinkProps extends LinkProps, StyleProps {
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
}

/** A router link styled as a button. */
export function ButtonLink({ variant, size, block, iconLeft, iconRight, className, children, ...rest }: ButtonLinkProps) {
  return (
    <Link className={buttonClass({ variant, size, block }, cx("btn--link", className))} {...rest}>
      {iconLeft}
      {children !== undefined && children !== null ? <span className="btn__label">{children}</span> : null}
      {iconRight}
    </Link>
  );
}
