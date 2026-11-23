import { useMemo, useState } from "react";
import { CircleCheck, CircleDashed, Download, Eye, Flag, Info, Lock, Pencil, ShieldCheck } from "lucide-react";
import { errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { ResultCell, ResultRow } from "../api/types";
import {
  Banner,
  Button,
  ButtonLink,
  Card,
  ConfirmDialog,
  CountBadge,
  DataTable,
  EmptyState,
  ErrorBanner,
  SkeletonTable,
  StatusPill,
  Tooltip,
  useToast,
  type Column,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { canEditExam, useExam } from "./ExamLayout";
import { StepFooter } from "./workflow";

const REVIEWABLE = new Set(["processed", "in_review"]);

function MarkCell({ cell }: { cell: ResultCell | null }) {
  if (!cell) return <span className="subtle" aria-label="No answer">-</span>;
  const pending = cell.status === "needs_review" || cell.status === "failed";
  const edited = Math.abs(cell.marks - cell.ai_marks) > 1e-6;
  return (
    <span className="mark-cell">
      <span className="num">{cell.marks}</span>
      {pending ? (
        <Tooltip content={cell.status === "failed" ? "Grading failed: needs review" : "Needs review"}>
          <span className="mark-cell__flag" tabIndex={0} aria-label="Needs review">
            <Flag size={12} aria-hidden="true" />
          </span>
        </Tooltip>
      ) : edited ? (
        <Tooltip content={`Changed by reviewer (AI: ${cell.ai_marks})`}>
          <span className="mark-cell__edited" tabIndex={0} aria-label={`Changed by reviewer, AI gave ${cell.ai_marks}`}>
            <Pencil size={11} aria-hidden="true" />
          </span>
        </Tooltip>
      ) : cell.status === "blank" ? (
        <Tooltip content="Blank answer: 0 marks">
          <span className="mark-cell__blank" tabIndex={0} aria-label="Blank answer">
            <CircleDashed size={11} aria-hidden="true" />
          </span>
        </Tooltip>
      ) : null}
    </span>
  );
}

/** One line of the readiness checklist. */
function Gate({ ok, label, detail }: { ok: boolean; label: string; detail: string }) {
  return (
    <li className={ok ? "gate is-ok" : "gate"}>
      <span className="gate__icon" aria-hidden="true">
        {ok ? <CircleCheck size={18} /> : <CircleDashed size={18} />}
      </span>
      <span className="gate__text">
        <span className="gate__label">{label}</span>
        <span className="gate__detail">{detail}</span>
      </span>
      <span className="sr-only">{ok ? "done" : "not yet"}</span>
    </li>
  );
}

/** Step 6: final approval. Approving a sheet locks its marks; only approved sheets are exported. */
export default function ApprovalPage() {
  const { exam, reload: reloadExam } = useExam();
  const canEdit = canEditExam(exam);
  const toast = useToast();
  const q = useAsync(() => api.getResults(exam.id), [exam.id]);
  const [exporting, setExporting] = useState(false);
  const [confirmApprove, setConfirmApprove] = useState(false);
  const [approving, setApproving] = useState(false);
  const [approvingOne, setApprovingOne] = useState<string | null>(null);

  const data = q.data;
  const rows = useMemo(() => data?.items ?? [], [data]);
  const reviewable = rows.filter((r) => REVIEWABLE.has(r.status));
  const pendingRows = reviewable.filter((r) => r.pending > 0);
  const approvable = reviewable.filter((r) => r.pending === 0);
  const approvedCount = rows.filter((r) => r.status === "approved").length;
  const evaluatedRows = rows.filter((r) => REVIEWABLE.has(r.status) || r.status === "approved");
  const inFlight = rows.filter((r) => r.status === "uploaded" || r.status === "processing").length;
  const failedCount = rows.filter((r) => r.status === "failed").length;
  const finalised = evaluatedRows.length > 0 && approvedCount === evaluatedRows.length && inFlight === 0;

  const exportCsv = async () => {
    setExporting(true);
    try {
      const name = await api.exportCsv(exam.id, `${exam.title.replace(/[^A-Za-z0-9_-]+/g, "_")}_results.csv`);
      toast.success("CSV downloaded", `${name}: ${approvedCount} approved sheet${approvedCount === 1 ? "" : "s"}.`);
    } catch (err) {
      toast.error("Export failed", errorMessage(err));
    } finally {
      setExporting(false);
    }
  };

  const approveAll = async () => {
    setApproving(true);
    try {
      const r = await api.approveAll(exam.id);
      const blocked = r.blocked.length;
      if (blocked) toast.info(`${r.approved.length} approved, ${blocked} blocked`, `Still pending review: ${r.blocked.map((b) => b.student_ref).join(", ")}`);
      else toast.success(`${r.approved.length} sheet${r.approved.length === 1 ? "" : "s"} approved`, "Approved marks are now locked and included in the CSV export.");
      setConfirmApprove(false);
      await q.reload(true);
      void reloadExam(true);
    } catch (err) {
      toast.error("Approve all failed", errorMessage(err));
      setConfirmApprove(false);
    } finally {
      setApproving(false);
    }
  };

  const approveOne = async (r: ResultRow) => {
    setApprovingOne(r.submission_id);
    try {
      await api.approveSubmission(r.submission_id);
      toast.success(`${r.student_ref} approved`, "Marks are locked and included in the CSV export.");
      await q.reload(true);
      void reloadExam(true);
    } catch (err) {
      toast.error(`Could not approve ${r.student_ref}`, errorMessage(err));
    } finally {
      setApprovingOne(null);
    }
  };

  const columns = useMemo<Column<ResultRow>[]>(() => {
    if (!data) return [];
    return [
      {
        key: "student",
        header: "Student",
        rowHeader: true,
        sortValue: (r) => r.student_ref,
        render: (r) => <span className="cell-strong">{r.student_ref}</span>,
      },
      ...data.questions.map<Column<ResultRow>>((qq, i) => ({
        key: `q${qq.question_no}`,
        header: (
          <span className="q-head">
            Q{qq.question_no}
            <span className="q-head__max">/{qq.max_marks}</span>
          </span>
        ),
        headerText: `Question ${qq.question_no}`,
        align: "right",
        sortValue: (r) => r.cells[i]?.marks ?? null,
        render: (r) => <MarkCell cell={r.cells[i]} />,
      })),
      {
        key: "total",
        header: (
          <span className="q-head">
            Total<span className="q-head__max">/{data.max_total}</span>
          </span>
        ),
        align: "right",
        sortValue: (r) => r.total,
        render: (r) => (r.total === null ? <span className="subtle">-</span> : <strong className="num">{r.total}</strong>),
      },
      {
        key: "pct",
        header: "%",
        align: "right",
        sortValue: (r) => (r.total !== null && data.max_total ? r.total / data.max_total : null),
        render: (r) => (r.total !== null && data.max_total ? <span className="num muted">{Math.round((r.total / data.max_total) * 100)}%</span> : <span className="subtle">-</span>),
      },
      {
        key: "status",
        header: "Status",
        sortValue: (r) => r.status,
        render: (r) => (
          <span className="row-sm">
            <StatusPill kind="submission" status={r.status} size="sm" />
            {r.pending ? <CountBadge count={r.pending} tone="warning" label="pending" /> : null}
          </span>
        ),
      },
      {
        key: "open",
        header: <span className="sr-only">Actions</span>,
        align: "right",
        render: (r) =>
          r.status === "uploaded" || r.status === "processing" || r.status === "failed" ? null : (
            <div className="row-actions">
              {REVIEWABLE.has(r.status) && canEdit ? (
                r.pending ? (
                  <ButtonLink size="sm" variant="subtle" to={`/exams/${exam.id}/review/${r.submission_id}`} iconLeft={<Flag size={14} aria-hidden="true" />}>
                    Review {r.pending}
                  </ButtonLink>
                ) : (
                  <Button size="sm" variant="secondary" loading={approvingOne === r.submission_id} disabled={approvingOne !== null} iconLeft={<ShieldCheck size={14} aria-hidden="true" />} onClick={() => void approveOne(r)}>
                    Approve
                  </Button>
                )
              ) : null}
              {r.status === "approved" ? (
                <span className="locked-chip">
                  <Lock size={12} aria-hidden="true" /> Locked
                </span>
              ) : null}
              <ButtonLink size="sm" variant="ghost" to={`/exams/${exam.id}/review/${r.submission_id}`} iconLeft={<Eye size={14} aria-hidden="true" />} aria-label={`View ${r.student_ref}`}>
                View
              </ButtonLink>
            </div>
          ),
      },
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, exam.id, canEdit, approvingOne]);

  if (q.loading && !data) {
    return (
      <Card flush>
        <SkeletonTable rows={6} cols={7} />
      </Card>
    );
  }
  if (q.error && !data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load results" />;
  if (!data || rows.length === 0) {
    return (
      <Card>
        <EmptyState illustration="results" title="Nothing to approve yet" description="Sheets appear here once the AI has marked them and you have reviewed them." />
      </Card>
    );
  }

  // Approve all approves every fully reviewed sheet and skips (reports) the rest, so it is
  // available whenever at least one sheet can be approved.
  const pendingNote =
    pendingRows.length > 0
      ? `${pendingRows.length} sheet${pendingRows.length === 1 ? " has" : "s have"} answers still needing review and will be skipped.`
      : null;
  const approveDisabledReason = !canEdit
    ? "Only the exam's owner can approve its sheets."
    : reviewable.length === 0
      ? "Nothing to approve: no processed sheets awaiting approval."
      : approvable.length === 0
        ? `Every sheet still has answers needing review (${pendingRows.length}). Review them first.`
        : null;
  const approveTooltip =
    approveDisabledReason ?? `Approve ${approvable.length} fully reviewed sheet${approvable.length === 1 ? "" : "s"}${pendingNote ? `; ${pendingNote}` : "."}`;

  return (
    <div className="stack">
      {finalised ? (
        <Banner
          tone="success"
          title="Results are final"
          action={
            <Button size="sm" variant="primary" iconLeft={<Download size={14} aria-hidden="true" />} loading={exporting} onClick={exportCsv}>
              Export CSV
            </Button>
          }
        >
          All {approvedCount} evaluated sheet{approvedCount === 1 ? " is" : "s are"} approved and locked. Reopen a sheet from its review page if a mark must change.
        </Banner>
      ) : null}
      <Card title="Ready for final approval?" subtitle="Approving locks a sheet's marks. Only approved sheets are exported.">
        <ul className="gates">
          <Gate
            ok={inFlight === 0 && evaluatedRows.length > 0}
            label="AI evaluation finished"
            detail={inFlight ? `${inFlight} sheet${inFlight === 1 ? " is" : "s are"} still being checked by the AI.` : `${evaluatedRows.length} sheet${evaluatedRows.length === 1 ? "" : "s"} marked by the AI.`}
          />
          <Gate
            ok={pendingRows.length === 0 && evaluatedRows.length > 0}
            label="Manual review complete"
            detail={pendingRows.length ? `${pendingRows.length} sheet${pendingRows.length === 1 ? " has" : "s have"} answers still to check: ${pendingRows.map((r) => r.student_ref).join(", ")}.` : "No answers are waiting for a teacher."}
          />
          <Gate
            ok={finalised}
            label="All sheets approved"
            detail={`${approvedCount} of ${evaluatedRows.length} approved${failedCount ? ` · ${failedCount} failed sheet${failedCount === 1 ? "" : "s"} not included` : ""}.`}
          />
        </ul>
      </Card>
      <Card
        title="Final marks"
        subtitle={`${rows.length} sheet${rows.length === 1 ? "" : "s"} · ${data.questions.length} question${data.questions.length === 1 ? "" : "s"} · out of ${data.max_total}`}
        flush
        actions={
          <>
            <Tooltip content={approveTooltip}>
              <span>
                <Button
                  variant="secondary"
                  iconLeft={<ShieldCheck size={16} aria-hidden="true" />}
                  disabled={!!approveDisabledReason}
                  onClick={() => setConfirmApprove(true)}
                >
                  {approvable.length && !approveDisabledReason ? `Approve all (${approvable.length})` : "Approve all"}
                </Button>
              </span>
            </Tooltip>
            <Button variant="primary" iconLeft={<Download size={16} aria-hidden="true" />} loading={exporting} disabled={approvedCount === 0} onClick={exportCsv}>
              Export CSV
            </Button>
          </>
        }
      >
        <DataTable
          caption={`Marks by student for ${exam.title}`}
          columns={columns}
          rows={rows}
          rowKey={(r) => r.submission_id}
          initialSort={{ key: "student", dir: "asc" }}
          maxHeight="65vh"
          dense
        />
      </Card>
      <StepFooter exam={exam} slug="approval" />
      <p className="legend-note">
        <Info size={14} aria-hidden="true" />
        <span>
          <Flag size={12} aria-hidden="true" className="inline-icon text-warning" /> needs review ·{" "}
          <Pencil size={11} aria-hidden="true" className="inline-icon" /> changed by a reviewer ·{" "}
          <CircleDashed size={11} aria-hidden="true" className="inline-icon" /> blank (0 marks)
        </span>
      </p>

      <ConfirmDialog
        open={confirmApprove}
        title="Approve all reviewed sheets?"
        message={
          <>
            {approvable.length} sheet{approvable.length === 1 ? "" : "s"} with no pending answers will be approved and locked.
            {pendingRows.length
              ? ` ${pendingRows.length} sheet${pendingRows.length === 1 ? "" : "s"} still needing review (${pendingRows.map((r) => r.student_ref).join(", ")}) will be skipped.`
              : ""}
          </>
        }
        confirmLabel={`Approve ${approvable.length}`}
        busy={approving}
        onConfirm={approveAll}
        onCancel={() => setConfirmApprove(false)}
      />
    </div>
  );
}
