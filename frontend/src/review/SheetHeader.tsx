// Sticky-feeling header of the review workspace: student sheet identity, totals, pending
// count, sheet switcher and the sheet-level actions (approve / reopen / next sheet).
import type { ReactNode, Ref } from "react";
import { ArrowLeft, ArrowRight, Flag, LockOpen, ShieldCheck, UserRound } from "lucide-react";
import type { SubmissionOut } from "../api/types";
import { Badge, Button, Select, StatusPill, Tooltip } from "../components";
import { fmtMarks } from "./flags";

interface SheetHeaderProps {
  sub: SubmissionOut;
  sheets: SubmissionOut[];
  maxTotal: number;
  pendingNos: number[];
  canEdit: boolean;
  sheetBusy: "approve" | "reopen" | null;
  approveRef: Ref<HTMLButtonElement>;
  nextSheet: SubmissionOut | null;
  shortcuts: ReactNode;
  onBack: () => void;
  onPickSheet: (id: string) => void;
  onApprove: () => void;
  onJumpPending: () => void;
  onReopen: () => void;
  onNextSheet: () => void;
}

function sheetOptionLabel(s: SubmissionOut): string {
  if (s.status === "approved") return `${s.student_ref} · approved`;
  if (s.status === "failed") return `${s.student_ref} · failed`;
  if (s.status === "uploaded" || s.status === "processing") return `${s.student_ref} · processing`;
  return s.answers_needing_review ? `${s.student_ref} · ${s.answers_needing_review} to review` : `${s.student_ref} · ready to approve`;
}

export function SheetHeader({
  sub,
  sheets,
  maxTotal,
  pendingNos,
  canEdit,
  sheetBusy,
  approveRef,
  nextSheet,
  shortcuts,
  onBack,
  onPickSheet,
  onApprove,
  onJumpPending,
  onReopen,
  onNextSheet,
}: SheetHeaderProps) {
  const approved = sub.status === "approved";
  const reviewable = sub.status === "processed" || sub.status === "in_review";
  const pending = pendingNos.length;
  const blockedText = `Review ${pending} more answer${pending === 1 ? "" : "s"} first: ${pendingNos.map((n) => `Q${n}`).join(", ")}`;
  const changed =
    sub.total_ai_marks !== null && sub.total_final_marks !== null && Math.abs(sub.total_ai_marks - sub.total_final_marks) > 1e-9;
  const options = sheets.some((s) => s.id === sub.id) ? sheets : [sub, ...sheets];

  return (
    <header className="sheet-bar">
      <div className="sheet-bar__id">
        <Button size="sm" variant="ghost" iconLeft={<ArrowLeft size={14} aria-hidden="true" />} onClick={onBack}>
          Queue
        </Button>
        <span className="sheet-bar__divider" aria-hidden="true" />
        <span className="sheet-bar__avatar" aria-hidden="true">
          <UserRound size={18} />
        </span>
        <div className="sheet-bar__text">
          <div className="row-sm">
            <h2 className="sheet-bar__title truncate">{sub.student_ref}</h2>
            <StatusPill kind="submission" status={sub.status} size="sm" />
          </div>
          <p className="sheet-bar__meta truncate">
            {sub.original_filename ? <span title={sub.original_filename}>{sub.original_filename}</span> : null}
            {sub.original_filename ? " · " : null}
            <span className="num">{sub.page_count}</span> page{sub.page_count === 1 ? "" : "s"}
          </p>
        </div>
      </div>

      <dl className="sheet-bar__totals">
        <div className="total">
          <dt className="total__label">AI total</dt>
          <dd className="total__value num">
            {fmtMarks(sub.total_ai_marks)}
            <small> / {fmtMarks(maxTotal)}</small>
          </dd>
        </div>
        <div className="total">
          <dt className="total__label">{approved ? "Final total" : "Current total"}</dt>
          <dd className="total__value num">
            {fmtMarks(sub.total_final_marks)}
            <small> / {fmtMarks(maxTotal)}</small>
            {changed ? <span className="sr-only"> (changed by the teacher)</span> : null}
          </dd>
        </div>
        <div className="total">
          <dt className="total__label">To review</dt>
          <dd className="total__value">
            {approved ? (
              <Badge size="sm" tone="success" icon={<ShieldCheck size={12} aria-hidden="true" />}>
                Locked
              </Badge>
            ) : pending ? (
              <Badge size="sm" tone="warning" icon={<Flag size={12} aria-hidden="true" />}>
                {pending} answer{pending === 1 ? "" : "s"}
              </Badge>
            ) : (
              <Badge size="sm" tone="success" icon={<ShieldCheck size={12} aria-hidden="true" />}>
                None
              </Badge>
            )}
          </dd>
        </div>
      </dl>

      <div className="sheet-bar__actions">
        <label className="sr-only" htmlFor="sheet-picker">
          Switch student sheet
        </label>
        <Select id="sheet-picker" inputSize="sm" className="sheet-bar__picker" value={sub.id} onChange={(e) => onPickSheet(e.target.value)}>
          {options.map((s) => (
            <option key={s.id} value={s.id}>
              {sheetOptionLabel(s)}
            </option>
          ))}
        </Select>
        {shortcuts}
        {canEdit && approved ? (
          <Button size="sm" variant="secondary" iconLeft={<LockOpen size={14} aria-hidden="true" />} loading={sheetBusy === "reopen"} onClick={onReopen}>
            Reopen
          </Button>
        ) : null}
        {canEdit && reviewable ? (
          pending ? (
            <Tooltip content={blockedText} side="bottom">
              <Button
                ref={approveRef}
                size="sm"
                variant="primary"
                aria-disabled="true"
                iconLeft={<ShieldCheck size={14} aria-hidden="true" />}
                onClick={onJumpPending}
              >
                Approve sheet
              </Button>
            </Tooltip>
          ) : (
            <Button
              ref={approveRef}
              size="sm"
              variant="primary"
              iconLeft={<ShieldCheck size={14} aria-hidden="true" />}
              loading={sheetBusy === "approve"}
              onClick={onApprove}
            >
              Approve sheet
            </Button>
          )
        ) : null}
        {nextSheet && (approved || (reviewable && !pending)) ? (
          <Button size="sm" variant="ghost" iconRight={<ArrowRight size={14} aria-hidden="true" />} onClick={onNextSheet}>
            Next sheet
          </Button>
        ) : null}
      </div>
    </header>
  );
}
