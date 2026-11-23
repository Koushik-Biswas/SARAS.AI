import { useMemo } from "react";
import { ArrowRight, CircleCheck, CloudUpload, Eye, Flag } from "lucide-react";
import * as api from "../api/endpoints";
import type { SubmissionOut } from "../api/types";
import { Badge, ButtonLink, Card, CountBadge, DataTable, EmptyState, ErrorBanner, SkeletonTable, StatusPill, type Column } from "../components";
import { useAsync } from "../hooks/useAsync";
import { usePolling } from "../hooks/usePolling";
import { useExam } from "./ExamLayout";
import { StepFooter } from "./workflow";

const REVIEWABLE = new Set(["processed", "in_review", "approved"]);

/**
 * Review tab landing: every processed sheet with its pending-answer count, most urgent first.
 * Rendered by ReviewPage when no submissionId is in the URL.
 */
export function ReviewQueue({ examId }: { examId: string }) {
  const { exam } = useExam();
  const q = useAsync(() => api.listAllSubmissions(examId), [examId]);
  const all = useMemo(() => q.data ?? [], [q.data]);
  const sheets = useMemo(
    () =>
      all
        .filter((s) => REVIEWABLE.has(s.status))
        .sort(
          (a, b) =>
            Number(a.status === "approved") - Number(b.status === "approved") ||
            b.answers_needing_review - a.answers_needing_review ||
            a.student_ref.localeCompare(b.student_ref, undefined, { numeric: true }),
        ),
    [all],
  );
  const processing = all.filter((s) => s.status === "uploaded" || s.status === "processing").length;
  usePolling(() => q.reload(true), 4000, processing > 0);

  const pendingTotal = sheets.reduce((a, s) => a + (s.status === "approved" ? 0 : s.answers_needing_review), 0);
  const nextSheet = sheets.find((s) => s.status !== "approved" && s.answers_needing_review > 0) ?? sheets.find((s) => s.status !== "approved");

  const columns: Column<SubmissionOut>[] = [
    {
      key: "student",
      header: "Student",
      rowHeader: true,
      sortValue: (s) => s.student_ref,
      render: (s) => <span className="cell-strong">{s.student_ref}</span>,
    },
    { key: "status", header: "Status", sortValue: (s) => s.status, render: (s) => <StatusPill kind="submission" status={s.status} size="sm" /> },
    {
      key: "pending",
      header: "Needs review",
      sortValue: (s) => (s.status === "approved" ? -1 : s.answers_needing_review),
      render: (s) =>
        s.status === "approved" ? (
          <span className="subtle">Locked</span>
        ) : s.answers_needing_review ? (
          <Badge tone="warning" size="sm" icon={<Flag size={12} aria-hidden="true" />}>
            {s.answers_needing_review} answer{s.answers_needing_review === 1 ? "" : "s"}
          </Badge>
        ) : (
          <Badge tone="success" size="sm" icon={<CircleCheck size={12} aria-hidden="true" />}>
            Ready to approve
          </Badge>
        ),
    },
    {
      key: "marks",
      header: "Marks",
      align: "right",
      sortValue: (s) => s.total_final_marks ?? s.total_ai_marks,
      render: (s) => <span className="num">{s.total_final_marks ?? s.total_ai_marks ?? "-"}</span>,
    },
    {
      key: "open",
      header: <span className="sr-only">Open</span>,
      align: "right",
      render: (s) => (
        <ButtonLink
          size="sm"
          variant={s.status === "approved" ? "ghost" : "subtle"}
          to={`/exams/${examId}/review/${s.id}`}
          iconLeft={<Eye size={14} aria-hidden="true" />}
        >
          {s.status === "approved" ? "View" : "Review"}
        </ButtonLink>
      ),
    },
  ];

  if (q.loading && !q.data) {
    return (
      <Card flush>
        <SkeletonTable rows={5} cols={5} />
      </Card>
    );
  }
  if (q.error && !q.data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load the review queue" />;
  if (sheets.length === 0) {
    return (
      <Card>
        <EmptyState
          illustration="review"
          title={processing ? "Sheets are still being processed" : "Nothing to review yet"}
          description={
            processing
              ? `${processing} sheet${processing === 1 ? " is" : "s are"} being read and graded. They'll appear here when ready.`
              : "Once uploaded sheets are processed, answers that need a human check are queued here."
          }
          action={
            processing ? (
              <ButtonLink to={`/exams/${examId}/status`} variant="secondary">
                View status
              </ButtonLink>
            ) : (
              <ButtonLink to={`/exams/${examId}/upload`} variant="primary" iconLeft={<CloudUpload size={16} aria-hidden="true" />}>
                Upload sheets
              </ButtonLink>
            )
          }
        />
      </Card>
    );
  }

  return (
    <div className="stack">
    <Card
      title="Review queue"
      subtitle={
        pendingTotal
          ? `${pendingTotal} answer${pendingTotal === 1 ? "" : "s"} across ${sheets.filter((s) => s.status !== "approved" && s.answers_needing_review).length} sheet(s) need a look.`
          : "All answers are reviewed. Approve sheets to lock their marks."
      }
      actions={
        nextSheet ? (
          <ButtonLink to={`/exams/${examId}/review/${nextSheet.id}`} variant="primary" iconRight={<ArrowRight size={16} aria-hidden="true" />}>
            {pendingTotal ? "Start reviewing" : "Open next sheet"}
            <CountBadge count={pendingTotal} tone="neutral" />
          </ButtonLink>
        ) : undefined
      }
      flush
    >
      <DataTable caption="Sheets to review" columns={columns} rows={sheets} rowKey={(s) => s.id} maxHeight="65vh" />
    </Card>
    <StepFooter
      exam={exam}
      slug="review"
      ready={pendingTotal === 0}
      hint={pendingTotal ? `${pendingTotal} answer${pendingTotal === 1 ? "" : "s"} still to check` : "Every answer is checked"}
    />
    </div>
  );
}
