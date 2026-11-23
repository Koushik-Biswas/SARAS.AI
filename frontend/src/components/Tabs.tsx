import { useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { cx } from "./cx";

/** Measures the active element inside `containerRef` and returns indicator geometry. */
function useIndicator(activeKey: string | undefined, deps: unknown[]) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<{ left: number; width: number } | null>(null);

  useLayoutEffect(() => {
    const measure = () => {
      const el = containerRef.current?.querySelector<HTMLElement>('[data-active="true"]');
      setRect(el ? { left: el.offsetLeft, width: el.offsetWidth } : null);
    };
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    if (ro && containerRef.current) ro.observe(containerRef.current);
    return () => ro?.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeKey, ...deps]);

  return { containerRef, rect };
}

// ---------------------------------------------------------------- route tabs (navigation)
export interface NavTabItem {
  to: string;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
  /** Extra path prefixes that should also mark this tab active. */
  alsoMatch?: string[];
}

/** Pill tab bar for route navigation, with an animated active indicator. */
export function NavTabs({ items, label, className }: { items: NavTabItem[]; label: string; className?: string }) {
  const { pathname } = useLocation();
  const matches = (p: string) => pathname === p || pathname.startsWith(`${p}/`);
  const active = items.find((i) => matches(i.to) || (i.alsoMatch ?? []).some(matches))?.to;
  const { containerRef, rect } = useIndicator(active, [items.length, items.map((i) => String(i.badge ?? "")).join()]);
  return (
    <nav aria-label={label} className={cx("tabs", className)}>
      <div ref={containerRef} className="tabs__list">
        {rect ? (
          <span className="tabs__indicator" aria-hidden="true" style={{ transform: `translateX(${rect.left}px)`, width: rect.width }} />
        ) : null}
        {items.map((i) => {
          const isActive = i.to === active;
          return (
            <NavLink
              key={i.to}
              to={i.to}
              className={cx("tabs__tab", isActive && "is-active")}
              data-active={isActive ? "true" : undefined}
              aria-current={isActive ? "page" : undefined}
            >
              {i.icon}
              <span>{i.label}</span>
              {i.badge}
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}

// ---------------------------------------------------------------- in-page tabs (ARIA tablist)
export interface TabItem<T extends string> {
  value: T;
  label: string;
  icon?: ReactNode;
  badge?: ReactNode;
}

interface TabsProps<T extends string> {
  items: TabItem<T>[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  /** id prefix: tab ids are `${idPrefix}-tab-${value}`, panels should use `${idPrefix}-panel-${value}`. */
  idPrefix: string;
  className?: string;
}

/** In-page tabs (role=tablist) with the same pill look; arrow keys move selection. */
export function Tabs<T extends string>({ items, value, onChange, label, idPrefix, className }: TabsProps<T>) {
  const { containerRef, rect } = useIndicator(value, [items.length]);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const idx = Math.max(0, items.findIndex((i) => i.value === value));
  const onKey = (e: KeyboardEvent) => {
    let next = -1;
    if (e.key === "ArrowRight") next = (idx + 1) % items.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + items.length) % items.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = items.length - 1;
    if (next >= 0) {
      e.preventDefault();
      onChange(items[next].value);
      refs.current[next]?.focus();
    }
  };
  return (
    <div className={cx("tabs", className)}>
      <div ref={containerRef} className="tabs__list" role="tablist" aria-label={label} onKeyDown={onKey}>
        {rect ? (
          <span className="tabs__indicator" aria-hidden="true" style={{ transform: `translateX(${rect.left}px)`, width: rect.width }} />
        ) : null}
        {items.map((i, n) => {
          const selected = i.value === value;
          return (
            <button
              key={i.value}
              ref={(el) => {
                refs.current[n] = el;
              }}
              type="button"
              role="tab"
              id={`${idPrefix}-tab-${i.value}`}
              aria-controls={`${idPrefix}-panel-${i.value}`}
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              data-active={selected ? "true" : undefined}
              className={cx("tabs__tab", selected && "is-active")}
              onClick={() => onChange(i.value)}
            >
              {i.icon}
              <span>{i.label}</span>
              {i.badge}
            </button>
          );
        })}
      </div>
    </div>
  );
}
