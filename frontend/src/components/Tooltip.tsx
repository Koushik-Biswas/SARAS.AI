import { cloneElement, isValidElement, useId, type ReactElement, type ReactNode } from "react";
import { cx } from "./cx";

interface TooltipProps {
  content: ReactNode;
  children: ReactElement;
  side?: "top" | "bottom";
  className?: string;
}

/**
 * Lightweight tooltip shown on hover and keyboard focus (CSS driven). The trigger gets
 * aria-describedby so screen readers announce the text. Keep content short.
 */
export function Tooltip({ content, children, side = "top", className }: TooltipProps) {
  const id = useId();
  const child = isValidElement(children)
    ? cloneElement(children as ReactElement<Record<string, unknown>>, { "aria-describedby": id })
    : children;
  return (
    <span className={cx("tooltip-host", className)}>
      {child}
      <span role="tooltip" id={id} className={cx("tooltip", `tooltip--${side}`)}>
        {content}
      </span>
    </span>
  );
}
