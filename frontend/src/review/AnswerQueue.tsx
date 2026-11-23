// Answer queue for one sheet: flagged answers first (server priority order), filterable
// between "Needs review" and "All". Each chip shows icon + question + confidence.
import { useEffect, useRef } from "react";
import type { AnswerOut } from "../api/types";
import { DelayedSpinner, ProgressBar, Segmented, cx, statusSpec } from "../components";
import { pct } from "./flags";

export type QueueFilter = "pending" | "all";

interface AnswerQueueProps {
  visible: AnswerOut[];
  total: number;
  pending: number;
  filter: QueueFilter;
  onFilter: (f: QueueFilter) => void;
  currentId: string | null;
  onSelect: (id: string) => void;
  /** A background refresh is running (spinner shows only after 300 ms). */
  refreshing?: boolean;
}

export function AnswerQueue({ visible, total, pending, filter, onFilter, currentId, onSelect, refreshing = false }: AnswerQueueProps) {
  const listRef = useRef<HTMLOListElement>(null);

  // Keep the active chip visible inside the horizontal strip (without scrolling the page).
  useEffect(() => {
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!list || !el) return;
    const left = el.offsetLeft;
    const right = left + el.offsetWidth;
    if (left < list.scrollLeft + 8) list.scrollLeft = Math.max(0, left - 8);
    else if (right > list.scrollLeft + list.clientWidth - 8) list.scrollLeft = right - list.clientWidth + 8;
  }, [currentId, filter, visible.length]);

  const cleared = total - pending;
  return (
    <nav className="queue-bar" aria-label="Answers on this sheet">
      <Segmented<QueueFilter>
        size="sm"
        label="Which answers to list"
        value={filter}
        onChange={onFilter}
        options={[
          { value: "pending", label: `Needs review · ${pending}` },
          { value: "all", label: `All · ${total}` },
        ]}
      />
      {visible.length ? (
        <ol className="queue" ref={listRef}>
          {visible.map((a) => {
            const spec = statusSpec("answer", a.status);
            const Icon = spec.icon;
            const active = a.answer_id === currentId;
            return (
              <li key={a.answer_id}>
                <button
                  type="button"
                  className={cx("qchip", `qchip--${spec.tone}`, active && "is-active")}
                  aria-current={active ? "true" : undefined}
                  aria-label={`Question ${a.question_no}: ${spec.label}, confidence ${pct(a.confidence)}`}
                  title={`Q${a.question_no} · ${spec.label}`}
                  onClick={() => onSelect(a.answer_id)}
                >
                  <Icon size={14} aria-hidden="true" className="qchip__icon" />
                  <span className="qchip__no num">Q{a.question_no}</span>
                  <span className="qchip__conf num" aria-hidden="true">
                    {pct(a.confidence)}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="queue__empty">
          Nothing needs review on this sheet.{" "}
          <button type="button" className="link-button" onClick={() => onFilter("all")}>
            Show all answers
          </button>
        </p>
      )}
      <DelayedSpinner loading={refreshing} size={14} label="Refreshing" />
      <div className="queue-bar__progress">
        <ProgressBar
          size="sm"
          tone={pending === 0 ? "success" : "accent"}
          value={total ? cleared / total : 0}
          label="Answers cleared"
          valueText={`${cleared} of ${total} cleared`}
        />
        <span className="queue-bar__count num" aria-hidden="true">
          {cleared}/{total} cleared
        </span>
      </div>
    </nav>
  );
}
