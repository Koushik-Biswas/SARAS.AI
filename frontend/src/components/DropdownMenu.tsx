import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { cx } from "./cx";

interface TriggerProps {
  id: string;
  "aria-haspopup": "menu";
  "aria-expanded": boolean;
  "aria-controls": string;
  onClick: () => void;
  onKeyDown: (e: KeyboardEvent<HTMLElement>) => void;
}

interface DropdownMenuProps {
  /** Render the trigger; spread the given props onto a <button>. */
  trigger: (props: TriggerProps, open: boolean) => ReactNode;
  children: ReactNode | ((close: () => void) => ReactNode);
  align?: "start" | "end";
  placement?: "bottom" | "top";
  className?: string;
  label?: string;
}

/**
 * Menu button pattern: click / Enter / ArrowDown opens, arrow keys move between
 * [role=menuitem] / [role=menuitemradio], Esc closes and returns focus, outside click closes.
 */
export function DropdownMenu({ trigger, children, align = "end", placement = "bottom", className, label }: DropdownMenuProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerId = useId();
  const menuId = useId();

  const items = () =>
    Array.from(menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"], [role="menuitemradio"]') ?? []).filter(
      (el) => !el.hasAttribute("disabled"),
    );

  const close = useCallback((focusTrigger = true) => {
    setOpen(false);
    if (focusTrigger) document.getElementById(triggerId)?.focus();
  }, [triggerId]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    const raf = requestAnimationFrame(() => items()[0]?.focus());
    return () => {
      document.removeEventListener("mousedown", onDown);
      cancelAnimationFrame(raf);
    };
  }, [open]);

  const onMenuKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const list = items();
    const idx = list.indexOf(document.activeElement as HTMLElement);
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      list[(idx + 1) % list.length]?.focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      list[(idx - 1 + list.length) % list.length]?.focus();
    } else if (e.key === "Home") {
      e.preventDefault();
      list[0]?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      list[list.length - 1]?.focus();
    }
    // Tab is allowed so non-menuitem controls (e.g. the theme segmented control) are reachable;
    // the menu closes when focus leaves the dropdown (see onBlur below).
  };

  const triggerProps: TriggerProps = {
    id: triggerId,
    "aria-haspopup": "menu",
    "aria-expanded": open,
    "aria-controls": menuId,
    onClick: () => setOpen((o) => !o),
    onKeyDown: (e) => {
      if (e.key === "ArrowDown" && !open) {
        e.preventDefault();
        setOpen(true);
      }
    },
  };

  return (
    <div
      ref={rootRef}
      className={cx("dropdown", className)}
      onBlur={(e) => {
        if (open && !rootRef.current?.contains(e.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      {trigger(triggerProps, open)}
      {open ? (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          tabIndex={-1}
          aria-labelledby={label ? undefined : triggerId}
          aria-label={label}
          className={cx("menu", `menu--${align}`, `menu--${placement}`)}
          onKeyDown={onMenuKey}
        >
          {typeof children === "function" ? children(() => close()) : children}
        </div>
      ) : null}
    </div>
  );
}

interface MenuItemProps {
  icon?: ReactNode;
  children: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  hint?: ReactNode;
}

export function MenuItem({ icon, children, onSelect, danger, disabled, hint }: MenuItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      tabIndex={-1}
      disabled={disabled}
      className={cx("menu__item", danger && "menu__item--danger")}
      onClick={onSelect}
    >
      {icon ? (
        <span className="menu__icon" aria-hidden="true">
          {icon}
        </span>
      ) : null}
      <span className="grow">{children}</span>
      {hint ? <span className="menu__hint">{hint}</span> : null}
    </button>
  );
}

export function MenuSeparator() {
  return <div className="menu__sep" role="separator" />;
}

export function MenuLabel({ children }: { children: ReactNode }) {
  return <div className="menu__label">{children}</div>;
}
