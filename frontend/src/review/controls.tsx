// Workspace controls: resizable split pane, zoom stepper, keyboard-shortcut popover.
import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { GripVertical, Keyboard, X, ZoomIn, ZoomOut } from "lucide-react";
import { Button, IconButton, Kbd, cx } from "../components";
import { useLocalStorage } from "../hooks/useLocalStorage";

// ---------------------------------------------------------------- SplitPane
interface SplitPaneProps {
  left: ReactNode;
  right: ReactNode;
  /** localStorage key that remembers the split. */
  storageKey: string;
  initial?: number;
  min?: number;
  max?: number;
  label: string;
  className?: string;
}

/** Two panes with a draggable, keyboard-operable vertical splitter (double-click resets). */
export function SplitPane({ left, right, storageKey, initial = 0.5, min = 0.3, max = 0.7, label, className }: SplitPaneProps) {
  const [stored, setStored] = useLocalStorage<number>(storageKey, initial);
  const ratio = Number.isFinite(stored) ? Math.min(max, Math.max(min, stored)) : initial;
  const rootRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const clampR = (v: number) => Math.min(max, Math.max(min, v));

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragging || !rootRef.current) return;
    const r = rootRef.current.getBoundingClientRect();
    if (r.width > 0) setStored(clampR((e.clientX - r.left) / r.width));
  };
  const endDrag = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    setDragging(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    let next: number | null = null;
    if (e.key === "ArrowLeft") next = ratio - step;
    else if (e.key === "ArrowRight") next = ratio + step;
    else if (e.key === "Home") next = min;
    else if (e.key === "End") next = max;
    else if (e.key === "Enter") next = initial;
    if (next !== null) {
      e.preventDefault();
      e.stopPropagation();
      setStored(clampR(next));
    }
  };

  return (
    <div ref={rootRef} className={cx("split", dragging && "is-dragging", className)} style={{ ["--split" as string]: `${ratio * 100}%` } as CSSProperties}>
      <div className="split__pane">{left}</div>
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label={label}
        aria-valuemin={Math.round(min * 100)}
        aria-valuemax={Math.round(max * 100)}
        aria-valuenow={Math.round(ratio * 100)}
        aria-valuetext={`Left pane ${Math.round(ratio * 100)}% wide`}
        tabIndex={0}
        title="Drag to resize · double-click to reset"
        className="split__handle"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onDoubleClick={() => setStored(initial)}
        onKeyDown={onKeyDown}
      >
        <span className="split__grip" aria-hidden="true">
          <GripVertical size={12} />
        </span>
      </div>
      <div className="split__pane">{right}</div>
    </div>
  );
}

// ---------------------------------------------------------------- Zoom
export const ZOOM_STEPS = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5];

export function zoomStep(zoom: number, dir: 1 | -1): number {
  if (dir > 0) return ZOOM_STEPS.find((z) => z > zoom + 1e-6) ?? ZOOM_STEPS[ZOOM_STEPS.length - 1];
  return [...ZOOM_STEPS].reverse().find((z) => z < zoom - 1e-6) ?? ZOOM_STEPS[0];
}

interface ZoomControlsProps {
  zoom: number;
  onZoom: (z: number) => void;
  /** Extra preset (e.g. "Fit page" in the region editor). */
  extra?: ReactNode;
}

export function ZoomControls({ zoom, onZoom, extra }: ZoomControlsProps) {
  const label = Math.abs(zoom - 1) < 1e-6 ? "Fit" : `${Math.round(zoom * 100)}%`;
  return (
    <div className="zoom" role="group" aria-label="Zoom">
      <IconButton
        size="sm"
        label="Zoom out (−)"
        icon={<ZoomOut size={16} aria-hidden="true" />}
        onClick={() => onZoom(zoomStep(zoom, -1))}
        disabled={zoom <= ZOOM_STEPS[0] + 1e-6}
      />
      <button
        type="button"
        className="zoom__value num"
        onClick={() => onZoom(1)}
        title="Fit width (0)"
        aria-label={`Zoom ${Math.round(zoom * 100)} percent of fit width. Reset to fit width`}
      >
        {label}
      </button>
      <IconButton
        size="sm"
        label="Zoom in (+)"
        icon={<ZoomIn size={16} aria-hidden="true" />}
        onClick={() => onZoom(zoomStep(zoom, 1))}
        disabled={zoom >= ZOOM_STEPS[ZOOM_STEPS.length - 1] - 1e-6}
      />
      {extra}
    </div>
  );
}

// ---------------------------------------------------------------- Shortcut help
export interface ShortcutDef {
  label: string;
  /** Alternatives; each is a key combo, e.g. [["J"], ["↓"]] or [["Ctrl", "Enter"]]. */
  keys: string[][];
}

interface ShortcutHelpProps {
  shortcuts: ShortcutDef[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  note?: ReactNode;
}

/** Non-modal popover listing keyboard shortcuts; Esc or an outside click closes it. */
export function ShortcutHelp({ shortcuts, open, onOpenChange, note }: ShortcutHelpProps) {
  const id = useId();
  const hostRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const setOpen = useRef(onOpenChange);
  setOpen.current = onOpenChange;

  useEffect(() => {
    if (!open) return;
    const raf = requestAnimationFrame(() => panelRef.current?.focus());
    const onDown = (e: MouseEvent) => {
      if (hostRef.current && !hostRef.current.contains(e.target as Node)) setOpen.current(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setOpen.current(false);
        btnRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="popover-host" ref={hostRef}>
      <Button
        ref={btnRef}
        size="sm"
        variant="ghost"
        iconLeft={<Keyboard size={16} aria-hidden="true" />}
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        onClick={() => onOpenChange(!open)}
      >
        Shortcuts <Kbd>?</Kbd>
      </Button>
      {open ? (
        <div id={id} ref={panelRef} role="dialog" aria-label="Keyboard shortcuts" className="popover" tabIndex={-1}>
          <div className="popover__head">
            <h3 className="popover__title">Keyboard shortcuts</h3>
            <IconButton
              size="sm"
              label="Close shortcuts"
              icon={<X size={16} aria-hidden="true" />}
              onClick={() => {
                onOpenChange(false);
                btnRef.current?.focus();
              }}
            />
          </div>
          <dl className="shortcuts">
            {shortcuts.map((s) => (
              <div className="shortcuts__row" key={s.label}>
                <dt>{s.label}</dt>
                <dd>
                  {s.keys.map((combo, i) => (
                    <span key={combo.join("+")} className="shortcuts__combo">
                      {i > 0 ? <span className="shortcuts__or">or</span> : null}
                      {combo.map((k) => (
                        <Kbd key={k}>{k}</Kbd>
                      ))}
                    </span>
                  ))}
                </dd>
              </div>
            ))}
          </dl>
          {note ? <p className="popover__foot">{note}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
