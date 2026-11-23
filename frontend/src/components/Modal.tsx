import { useEffect, useId, useRef, useState, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";
import { TriangleExclamationPoint, X } from "lucide-react";
import { Button, type ButtonVariant } from "./Button";
import { Field, Textarea } from "./Field";
import { cx } from "./cx";

const FOCUSABLE_PARTS = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
];
const FOCUSABLE = FOCUSABLE_PARTS.join(", ");
/** Focusables inside the body only (each selector in the list must be prefixed). */
const BODY_FOCUSABLE = FOCUSABLE_PARTS.map((s) => `.modal__body ${s}`).join(", ");

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  size?: "sm" | "md" | "lg";
  /** Prevent closing via Esc / backdrop (e.g. while saving). */
  dismissible?: boolean;
  /** Element to focus first (defaults to the first focusable control). */
  initialFocusRef?: RefObject<HTMLElement>;
}

/** Accessible dialog: portal, focus trap, Esc to close, focus returns to the opener. */
export function Modal({ open, onClose, title, description, children, footer, size = "md", dismissible = true, initialFocusRef }: ModalProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const descId = useId();
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const dismissRef = useRef(dismissible);
  dismissRef.current = dismissible;

  useEffect(() => {
    if (!open) return;
    const opener = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    const focusFirst = () => {
      const target = initialFocusRef?.current ?? panel?.querySelector<HTMLElement>(BODY_FOCUSABLE) ?? panel?.querySelector<HTMLElement>(FOCUSABLE) ?? panel;
      target?.focus();
    };
    const raf = requestAnimationFrame(focusFirst);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && dismissRef.current) {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key === "Tab" && panel) {
        const nodes = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((n) => n.offsetParent !== null);
        if (nodes.length === 0) {
          e.preventDefault();
          return;
        }
        const first = nodes[0];
        const last = nodes[nodes.length - 1];
        if (e.shiftKey && (document.activeElement === first || document.activeElement === panel)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = prevOverflow;
      if (opener && document.contains(opener)) opener.focus();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;
  return createPortal(
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && dismissRef.current) onCloseRef.current();
      }}
    >
      <div
        ref={panelRef}
        className={cx("modal", `modal--${size}`)}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descId : undefined}
        tabIndex={-1}
      >
        <header className="modal__header">
          <div>
            <h2 id={titleId} className="modal__title">
              {title}
            </h2>
            {description ? (
              <p id={descId} className="modal__desc">
                {description}
              </p>
            ) : null}
          </div>
          {dismissible ? (
            <button type="button" className="modal__close" aria-label="Close dialog" onClick={() => onCloseRef.current()}>
              <X size={18} aria-hidden="true" />
            </button>
          ) : null}
        </header>
        {children ? <div className="modal__body">{children}</div> : null}
        {footer ? <footer className="modal__footer">{footer}</footer> : null}
      </div>
    </div>,
    document.body,
  );
}

interface ConfirmDialogProps {
  open: boolean;
  title: ReactNode;
  message?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: "danger" | "primary";
  busy?: boolean;
  /** Ask for a free-text reason (required when `reasonRequired`). */
  reasonLabel?: string;
  reasonRequired?: boolean;
  onConfirm: (reason: string) => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  tone = "primary",
  busy,
  reasonLabel,
  reasonRequired,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [reason, setReason] = useState("");
  const [touched, setTouched] = useState(false);
  useEffect(() => {
    if (open) {
      setReason("");
      setTouched(false);
    }
  }, [open]);
  const missing = !!reasonRequired && !reason.trim();
  const variant: ButtonVariant = tone === "danger" ? "danger" : "primary";
  return (
    <Modal
      open={open}
      onClose={onCancel}
      size="sm"
      dismissible={!busy}
      title={
        <span className="row-sm">
          {tone === "danger" ? (
            <span className="modal__danger-icon" aria-hidden="true">
              <TriangleExclamationPoint size={18} />
            </span>
          ) : null}
          {title}
        </span>
      }
      footer={
        <>
          <Button variant="ghost" onClick={onCancel} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button
            variant={variant}
            loading={busy}
            onClick={() => {
              setTouched(true);
              if (!missing) onConfirm(reason.trim());
            }}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      {message ? <div className="modal__message">{message}</div> : null}
      {reasonLabel ? (
        <Field label={reasonLabel} required={reasonRequired} error={touched && missing ? "Please give a reason." : undefined}>
          <Textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={3} maxLength={1000} />
        </Field>
      ) : null}
    </Modal>
  );
}
