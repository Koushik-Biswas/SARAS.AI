import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CircleCheck, CircleX, Info, X } from "lucide-react";
import { cx } from "./cx";

export type ToastTone = "success" | "error" | "info";

interface ToastItem {
  id: number;
  tone: ToastTone;
  title: string;
  message?: string;
  duration: number;
}

interface ToastApi {
  show: (t: { tone?: ToastTone; title: string; message?: string; duration?: number }) => void;
  success: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastApi | null>(null);
const ICONS = { success: CircleCheck, error: CircleX, info: Info };
const MAX_VISIBLE = 4;

function ToastView({ item, onDismiss }: { item: ToastItem; onDismiss: (id: number) => void }) {
  const [paused, setPaused] = useState(false);
  const remaining = useRef(item.duration);
  const started = useRef(Date.now());

  useEffect(() => {
    if (paused || item.duration <= 0) return;
    started.current = Date.now();
    const t = window.setTimeout(() => onDismiss(item.id), remaining.current);
    return () => {
      window.clearTimeout(t);
      remaining.current -= Date.now() - started.current;
    };
  }, [paused, item.id, item.duration, onDismiss]);

  const Icon = ICONS[item.tone];
  return (
    <div
      className={cx("toast", `toast--${item.tone}`)}
      role={item.tone === "error" ? "alert" : "status"}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <span className="toast__icon" aria-hidden="true">
        <Icon size={18} />
      </span>
      <div className="toast__content">
        <p className="toast__title">{item.title}</p>
        {item.message ? <p className="toast__message">{item.message}</p> : null}
      </div>
      <button type="button" className="toast__close" aria-label="Dismiss notification" onClick={() => onDismiss(item.id)}>
        <X size={16} aria-hidden="true" />
      </button>
    </div>
  );
}

/** Stacked bottom-right toasts in an aria-live="polite" region; auto-dismiss (paused on hover/focus). */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const show = useCallback<ToastApi["show"]>(({ tone = "info", title, message, duration }) => {
    const id = nextId.current++;
    const d = duration ?? (tone === "error" ? 7000 : 4500);
    setItems((xs) => [...xs, { id, tone, title, message, duration: d }].slice(-MAX_VISIBLE));
  }, []);

  const api = useMemo<ToastApi>(
    () => ({
      show,
      dismiss,
      success: (title, message) => show({ tone: "success", title, message }),
      error: (title, message) => show({ tone: "error", title, message }),
      info: (title, message) => show({ tone: "info", title, message }),
    }),
    [show, dismiss],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="toasts" aria-live="polite" aria-relevant="additions" aria-label="Notifications">
        {items.map((t) => (
          <ToastView key={t.id} item={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
  return ctx;
}
