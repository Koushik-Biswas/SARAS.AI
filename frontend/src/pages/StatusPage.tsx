import { useMemo, useState } from "react";
import { CircleX, CloudUpload, Ellipsis, Eye, RotateCcw, Trash } from "lucide-react";
import * as api from "../api/endpoints";
import type { SubmissionOut, SubmissionStatus } from "../api/types";
import {
  Button,
  ButtonLink,
  Card,
  ConfirmDialog,
  CountBadge,
  DataTable,
  DelayedSpinner,
  DropdownMenu,
  EmptyState,
  ErrorBanner,
  IconButton,
  MenuItem,
  ProgressBar,
  Segmented,
  SkeletonTable,
  StatusPill,
  Tooltip,
  type Column,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { usePolling } from "../hooks/usePolling";
import { canEditExam, useExam } from "./ExamLayout";
import { stageLabel } from "./stages";
import { AiEnginePanel } from "./AiEnginePanel";
import { PipelineMonitor } from "./PipelineMonitor";
import { StepFooter } from "./workflow";

const ACTIVE: SubmissionStatus[] = ["uploaded", "processing"];
const REVIEWABLE: SubmissionStatus[] = ["processed", "in_review", "approved"];

type Filter = "all" | "active" | "ready" | "failed" | "approved";

function matches(s: SubmissionOut, f: Filter): boolean {
  if (f === "all") return true;
  if (f === "active") return ACTIVE.includes(s.status);
  if (f === "ready") return s.status === "processed" || s.status === "in_review";
  if (f === "failed") return s.status === "failed";
  return s.status === "approved";
}

export default function StatusPage() {
  const { exam, reload: reloadExam } = useExam();
  const canEdit = canEditExam(exam);
  const q = useAsync((): Promise<SubmissionOut[]> => api.listAllSubmissions(exam.id), [exam.id]);
  const subs = useMemo(() => q.data ?? [], [q.data]);
  const [filter, setFilter] = useState<Filter>("all");
  const [retrying, setRetrying] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<SubmissionOut | null>(null);
  const [deleting, setDeleting] = useState(false);
  // The failed action itself, so the banner's Retry really runs it again.
  const [actionError, setActionError] = useState<{ error: unknown; title: string; again: () => void } | null>(null);

  const anyActive = subs.some((s) => ACTIVE.includes(s.status));
  usePolling(
    async () => {
      const wasActive = subs.filter((s) => ACTIVE.includes(s.status)).length;
      await q.reload(true);
      if (wasActive) void reloadExam(true);
    },
    1500,
    anyActive,
  );

  const counts = useMemo(
    () => ({
      all: subs.length,
      active: subs.filter((s) => matches(s, "active")).length,
      ready: subs.filter((s) => matches(s, "ready")).length,
      failed: subs.filter((s) => matches(s, "failed")).length,
      approved: subs.filter((s) => matches(s, "approved")).length,
    }),
    [subs],
  );
  const rows = subs.filter((s) => matches(s, filter));

  const retry = async (s: SubmissionOut) => {
    setRetrying(s.id);
    setActionError(null);
    try {
      const updated = await api.retrySubmission(s.id);
      q.setData((prev) => (prev ?? []).map((x) => (x.id === s.id ? updated : x)));
      void reloadExam(true);
    } catch (err) {
      setActionError({ error: err, title: `Could not re-queue ${s.student_ref}`, again: () => void retry(s) });
    } finally {
      setRetrying(null);
    }
  };

  const remove = async (target: SubmissionOut) => {
    setDeleting(true);
    setActionError(null);
    try {
      await api.deleteSubmission(target.id);
      q.setData((prev) => (prev ?? []).filter((x) => x.id !== target.id));
      void reloadExam(true);
    } catch (err) {
      setActionError({ error: err, title: `Could not delete ${target.student_ref}`, again: () => void remove(target) });
    } finally {
      setToDelete(null);
      setDeleting(false);
    }
  };

  const columns: Column<SubmissionOut>[] = [
    {
      key: "student",
      header: "Student",
      rowHeader: true,
      sortValue: (s) => s.student_ref,
      render: (s) => (
        <div className="cell-stack">
          <span className="cell-strong">{s.student_ref}</span>
          {s.original_filename ? <span className="cell-sub truncate" title={s.original_filename}>{s.original_filename}</span> : null}
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortValue: (s) => s.status,
      render: (s) => <StatusPill kind="submission" status={s.status} size="sm" />,
    },
    {
      key: "stage",
      header: "Stage",
      render: (s) => <span className={s.status === "failed" ? "text-danger" : "muted"}>{stageLabel(s)}</span>,
    },
    {
      key: "pages",
      header: "Pages",
      width: 170,
      sortValue: (s) => s.page_count,
      render: (s) =>
        s.page_count ? (
          <ProgressBar
            size="sm"
            tone={s.status === "failed" ? "danger" : s.pages_done >= s.page_count ? "success" : "accent"}
            label={`${s.student_ref}: pages processed`}
            showLabel
            value={s.pages_done / s.page_count}
            valueText={`${s.pages_done} / ${s.page_count}`}
          />
        ) : (
          <span className="subtle">-</span>
        ),
    },
    {
      key: "review",
      header: "To review",
      align: "right",
      sortValue: (s) => s.answers_needing_review,
      render: (s) =>
        REVIEWABLE.includes(s.status) ? (
          s.answers_needing_review ? (
            <CountBadge count={s.answers_needing_review} tone="warning" label="answers need review" />
          ) : (
            <span className="subtle">0</span>
          )
        ) : (
          <span className="subtle">-</span>
        ),
    },
    {
      key: "marks",
      header: "Marks",
      align: "right",
      sortValue: (s) => s.total_final_marks ?? s.total_ai_marks,
      render: (s) => {
        const m = s.total_final_marks ?? s.total_ai_marks;
        return m === null || m === undefined ? <span className="subtle">-</span> : <span className="num">{m}</span>;
      },
    },
    {
      key: "issues",
      header: "Issues",
      render: (s) => {
        const pageIssues = Object.keys(s.page_flags ?? {}).length;
        if (s.error) {
          return (
            <Tooltip content={s.error}>
              <span className="issue-text" tabIndex={0}>
                <CircleX size={14} aria-hidden="true" /> <span className="truncate">{s.error}</span>
              </span>
            </Tooltip>
          );
        }
        if (pageIssues || s.flags.length) {
          return (
            <span className="muted">
              {pageIssues ? `${pageIssues} page${pageIssues === 1 ? "" : "s"} flagged` : s.flags.slice(0, 2).map((f) => f.replace(/_/g, " ")).join(", ")}
            </span>
          );
        }
        return <span className="subtle">None</span>;
      },
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "right",
      render: (s) => (
        <div className="row-actions">
          {s.status === "failed" && canEdit ? (
            <Button size="sm" variant="secondary" iconLeft={<RotateCcw size={14} aria-hidden="true" />} loading={retrying === s.id} onClick={() => retry(s)}>
              Retry
            </Button>
          ) : null}
          {REVIEWABLE.includes(s.status) ? (
            <ButtonLink size="sm" variant={s.status === "approved" ? "ghost" : "subtle"} to={`/exams/${exam.id}/review/${s.id}`} iconLeft={<Eye size={14} aria-hidden="true" />}>
              {s.status === "approved" ? "View" : "Review"}
            </ButtonLink>
          ) : null}
          {canEdit ? (
          <DropdownMenu
            label={`Actions for ${s.student_ref}`}
            trigger={(props) => <IconButton {...props} size="sm" label={`More actions for ${s.student_ref}`} icon={<Ellipsis size={16} aria-hidden="true" />} />}
          >
            {(close) => (
              <>
                {s.status === "processed" ? (
                  <MenuItem
                    icon={<RotateCcw size={16} />}
                    onSelect={() => {
                      close();
                      void retry(s);
                    }}
                  >
                    Reprocess sheet
                  </MenuItem>
                ) : null}
                <MenuItem
                  danger
                  icon={<Trash size={16} />}
                  disabled={s.status === "approved" || s.status === "processing"}
                  onSelect={() => {
                    close();
                    setToDelete(s);
                  }}
                >
                  {s.status === "processing" ? "Delete sheet (after processing)" : "Delete sheet"}
                </MenuItem>
              </>
            )}
          </DropdownMenu>
          ) : null}
        </div>
      ),
    },
  ];

  if (q.loading && !q.data) {
    return (
      <Card flush>
        <SkeletonTable rows={5} cols={6} />
      </Card>
    );
  }
  if (q.error && !q.data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load sheets" />;

  if (subs.length === 0) {
    return (
      <Card>
        <EmptyState
          illustration="upload"
          title="No sheets uploaded yet"
          description="Upload scanned answer sheets and their processing progress will appear here."
          action={
            canEdit ? (
              <ButtonLink to={`/exams/${exam.id}/upload`} variant="primary" iconLeft={<CloudUpload size={16} aria-hidden="true" />}>
                Upload sheets
              </ButtonLink>
            ) : undefined
          }
        />
      </Card>
    );
  }

  return (
    <div className="stack">
      {actionError ? (
        <ErrorBanner
          error={actionError.error}
          title={actionError.title}
          retrying={retrying !== null || deleting}
          onRetry={actionError.again}
        />
      ) : null}
      {q.error ? <ErrorBanner error={q.error} title="Live updates paused" onRetry={() => q.reload(true)} /> : null}
      <AiEnginePanel exam={exam} subs={subs} />
      <PipelineMonitor subs={subs} />
      <Card
        title="Sheet by sheet"
        subtitle={anyActive ? "Live: updated from the engine every 1.5 seconds." : `${counts.all} sheet${counts.all === 1 ? "" : "s"}.`}
        actions={
          <div className="row-sm">
            <DelayedSpinner loading={anyActive} label="Live" size={14} />
            <Segmented<Filter>
              label="Filter sheets"
              size="sm"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: `All ${counts.all}` },
                { value: "active", label: `In progress ${counts.active}` },
                { value: "ready", label: `To review ${counts.ready}` },
                { value: "failed", label: `Failed ${counts.failed}` },
                { value: "approved", label: `Approved ${counts.approved}` },
              ]}
            />
          </div>
        }
        flush
      >
        <DataTable
          caption="Uploaded answer sheets and their processing status"
          columns={columns}
          rows={rows}
          rowKey={(s) => s.id}
          initialSort={{ key: "student", dir: "asc" }}
          maxHeight="65vh"
          empty={<span className="subtle">No sheets match this filter.</span>}
        />
      </Card>
      <StepFooter
        exam={exam}
        slug="status"
        ready={!anyActive && counts.all - counts.failed > 0}
        hint={anyActive ? "You can continue: results fill in as each sheet finishes" : undefined}
      />

      <ConfirmDialog
        open={!!toDelete}
        tone="danger"
        title="Delete this sheet?"
        message={
          <>
            The uploaded PDF for <strong>{toDelete?.student_ref}</strong> and all of its marks will be deleted. This cannot be undone.
          </>
        }
        confirmLabel="Delete sheet"
        busy={deleting}
        onConfirm={() => toDelete && void remove(toDelete)}
        onCancel={() => setToDelete(null)}
      />
    </div>
  );
}
