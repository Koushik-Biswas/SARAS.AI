import {
  cloneElement,
  forwardRef,
  isValidElement,
  useId,
  type InputHTMLAttributes,
  type ReactElement,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";
import { ChevronDown, CircleExclamationPoint } from "lucide-react";
import { cx } from "./cx";

interface FieldProps {
  label?: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  required?: boolean;
  /** Visually hide the label (still read by screen readers). */
  hideLabel?: boolean;
  className?: string;
  /** A single form control; it receives id / aria-describedby / aria-invalid. */
  children: ReactElement;
}

/** Label + control + hint + inline error, wired for assistive tech. */
export function Field({ label, hint, error, required, hideLabel, className, children }: FieldProps) {
  const auto = useId();
  const id = (isValidElement(children) && (children.props as { id?: string }).id) || auto;
  const hintId = hint ? `${id}-hint` : undefined;
  const errId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errId].filter(Boolean).join(" ") || undefined;
  const control = cloneElement(children as ReactElement<Record<string, unknown>>, {
    id,
    "aria-describedby": describedBy,
    "aria-invalid": error ? true : undefined,
    required,
  });
  return (
    <div className={cx("field", error ? "field--error" : undefined, className)}>
      {label ? (
        <label htmlFor={id} className={cx("field__label", hideLabel && "sr-only")}>
          {label}
          {required ? (
            <span className="field__req" aria-hidden="true">
              *
            </span>
          ) : null}
        </label>
      ) : null}
      {control}
      {hint ? (
        <p id={hintId} className={error ? "sr-only" : "field__hint"}>
          {hint}
        </p>
      ) : null}
      {error ? (
        <p id={errId} className="field__error">
          <CircleExclamationPoint size={14} aria-hidden="true" />
          <span>{error}</span>
        </p>
      ) : null}
    </div>
  );
}

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  iconLeft?: ReactNode;
  /** Element rendered inside the right edge (e.g. a show-password IconButton). */
  endAdornment?: ReactNode;
  inputSize?: "sm" | "md" | "lg";
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { iconLeft, endAdornment, inputSize = "md", className, ...rest },
  ref,
) {
  const input = <input ref={ref} className={cx("input", `input--${inputSize}`, className)} {...rest} />;
  if (!iconLeft && !endAdornment) return input;
  return (
    <div className={cx("input-wrap", iconLeft ? "has-icon" : undefined, endAdornment ? "has-end" : undefined)}>
      {iconLeft ? (
        <span className="input-wrap__icon" aria-hidden="true">
          {iconLeft}
        </span>
      ) : null}
      {input}
      {endAdornment ? <span className="input-wrap__end">{endAdornment}</span> : null}
    </div>
  );
});

export interface NumberInputProps extends Omit<InputProps, "type" | "value" | "onChange"> {
  value: number | null | undefined;
  onValueChange: (v: number | null) => void;
  /** Max decimals kept (marks: 2). */
  decimals?: number;
}

/** Numeric input: emits null when empty and rounds to `decimals` places. */
export const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(function NumberInput(
  { value, onValueChange, decimals = 2, className, step, ...rest },
  ref,
) {
  return (
    <Input
      ref={ref}
      type="number"
      inputMode="decimal"
      step={step ?? (decimals > 0 ? 1 / 10 ** Math.min(decimals, 2) : 1)}
      className={cx("input--number num", className)}
      value={value === null || value === undefined ? "" : value}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onValueChange(null);
        const n = Number(raw);
        if (!Number.isFinite(n)) return;
        const f = 10 ** decimals;
        onValueChange(Math.round(n * f) / f);
      }}
      {...rest}
    />
  );
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, rows = 3, ...rest },
  ref,
) {
  return <textarea ref={ref} rows={rows} className={cx("input", "textarea", className)} {...rest} />;
});

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  inputSize?: "sm" | "md" | "lg";
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, inputSize = "md", children, ...rest },
  ref,
) {
  return (
    <span className="select-wrap">
      <select ref={ref} className={cx("input", "select", `input--${inputSize}`, className)} {...rest}>
        {children}
      </select>
      <ChevronDown className="select-wrap__chevron" size={16} aria-hidden="true" />
    </span>
  );
});
