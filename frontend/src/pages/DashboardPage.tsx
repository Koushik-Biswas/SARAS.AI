import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ArrowRight, BookOpen, Calendar, ClipboardCheck, FileText, Flag, Plus, ShieldCheck, Table2 } from "lucide-react";
import type { ExamOut } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Badge, Button, ButtonLink, CountBadge, EmptyState, ErrorBanner, ProgressBar, Skeleton, StatTile } from "../components";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { useBreadcrumbs, useExamIndex } from "../layout/ShellContext";
import { ExamFormModal } from "./ExamFormModal";
import { formatDate } from "./ExamLayout";

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 17) return "Good afternoon";
  return "Good evening";
}

function ExamCard({ exam }: { exam: ExamOut }) {
  const s = exam.stats;
  const date = formatDate(exam.date);
  const approvedFrac = s.sheets ? s.approved / s.sheets : 0;
  const setupNeeded = s.questions === 0;
  return (
    <article className="exam-card">
      <div className="exam-card__top">
        <div className="exam-card__icon" aria-hidden="true">
          <BookOpen size={18} />
        </div>
        <div className="exam-card__heading">
          <h3 className="exam-card__title">
            <Link to={`/exams/${exam.id}`} className="exam-card__link">
              {exam.title}
            </Link>
          </h3>
          <p className="exam-card__meta">
            {exam.subject ? <span>{exam.subject}</span> : null}
            {date ? (
              <span className="row-sm">
                <Calendar size={13} aria-hidden="true" />
                {date}
              </span>
            ) : null}
            {!exam.subject && !date ? <span>{exam.total_marks} marks</span> : null}
          </p>
        </div>
        {setupNeeded ? (
          <Badge tone="accent" size="sm">
            Needs rubric
          </Badge>
        ) : s.needs_review ? (
          <Badge tone="warning" size="sm" icon={<Flag size={12} aria-hidden="true" />}>
            Review
          </Badge>
        ) : s.sheets && s.approved === s.sheets ? (
          <Badge tone="success" size="sm" icon={<ShieldCheck size={12} aria-hidden="true" />}>
            Done
          </Badge>
        ) : null}
      </div>

      <dl className="exam-card__stats">
        <div>
          <dt>Sheets processed</dt>
          <dd className="num">
            {s.processed}
            <span className="subtle"> / {s.sheets}</span>
          </dd>
        </div>
        <div>
          <dt>Need review</dt>
          <dd className="num">{s.needs_review}</dd>
        </div>
        <div>
          <dt>Average</dt>
          <dd className="num">{s.average_marks !== null ? s.average_marks : "-"}</dd>
        </div>
      </dl>

      <ProgressBar
        value={approvedFrac}
        size="sm"
        tone="success"
        label={`${exam.title}: approved sheets`}
        showLabel
        valueText={`${s.approved} of ${s.sheets} approved`}
      />

      <div className="exam-card__actions">
        {setupNeeded ? (
          <ButtonLink to={`/exams/${exam.id}/rubric`} size="sm" variant="subtle" iconRight={<ArrowRight size={14} aria-hidden="true" />}>
            Set up rubric
          </ButtonLink>
        ) : (
          <>
            <ButtonLink to={`/exams/${exam.id}/review`} size="sm" variant="subtle" iconLeft={<ClipboardCheck size={14} aria-hidden="true" />}>
              Review
              <CountBadge count={s.needs_review} tone="warning" label="pending" />
            </ButtonLink>
            <ButtonLink to={`/exams/${exam.id}/results`} size="sm" variant="ghost" iconLeft={<Table2 size={14} aria-hidden="true" />}>
              Results
            </ButtonLink>
          </>
        )}
      </div>
    </article>
  );
}

function ExamCardSkeleton() {
  return (
    <div className="exam-card" aria-hidden="true">
      <div className="exam-card__top">
        <Skeleton width={38} height={38} radius={10} />
        <div className="grow stack-sm">
          <Skeleton width="70%" height={16} />
          <Skeleton width="40%" height={12} />
        </div>
      </div>
      <Skeleton height={44} radius={10} />
      <Skeleton height={6} radius={99} />
      <div className="row-sm">
        <Skeleton width={90} height={30} radius={6} />
        <Skeleton width={80} height={30} radius={6} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  useDocumentTitle("Dashboard");
  useBreadcrumbs(1, null);
  const { user } = useAuth();
  const { exams, total, loading, error, reload, fetching, upsert } = useExamIndex();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [creating, setCreating] = useState(false);

  // Sidebar "New exam" links here with ?new=1.
  useEffect(() => {
    if (params.get("new") === "1") {
      setCreating(true);
      params.delete("new");
      setParams(params, { replace: true });
    }
  }, [params, setParams]);

  // Refresh stats when returning to the dashboard (skip if the first load is still running).
  useEffect(() => {
    if (!loading) void reload(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const totals = useMemo(
    () =>
      exams.reduce(
        (acc, e) => ({
          processed: acc.processed + e.stats.processed,
          sheets: acc.sheets + e.stats.sheets,
          review: acc.review + e.stats.needs_review,
          approved: acc.approved + e.stats.approved,
        }),
        { processed: 0, sheets: 0, review: 0, approved: 0 },
      ),
    [exams],
  );
  const firstName = user?.name.trim().split(/\s+/)[0] ?? "";
  const showSkeleton = loading && exams.length === 0;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1 className="page-header__title">
            {greeting()}
            {firstName ? `, ${firstName}` : ""}
          </h1>
          <p className="page-header__sub">
            {totals.review > 0
              ? `${totals.review} answer${totals.review === 1 ? "" : "s"} across your exams need${totals.review === 1 ? "s" : ""} a quick look.`
              : "Here's how your marking is going."}
          </p>
        </div>
        <div className="page-header__actions">
          <Button variant="primary" iconLeft={<Plus size={16} aria-hidden="true" />} onClick={() => setCreating(true)}>
            New exam
          </Button>
        </div>
      </header>

      <section className="stat-grid" aria-label="Summary">
        <StatTile label="Exams" value={total} icon={<BookOpen size={18} />} loading={showSkeleton} sub="Created by you" />
        <StatTile
          label="Sheets processed"
          value={
            <>
              {totals.processed}
              <small>/ {totals.sheets}</small>
            </>
          }
          icon={<FileText size={18} />}
          tone="info"
          loading={showSkeleton}
          sub="Read and graded"
        />
        <StatTile
          label="Answers needing review"
          value={totals.review}
          icon={<Flag size={18} />}
          tone="warning"
          loading={showSkeleton}
          sub="Low confidence or flagged"
        />
        <StatTile label="Sheets approved" value={totals.approved} icon={<ShieldCheck size={18} />} tone="success" loading={showSkeleton} sub="Ready to export" />
      </section>

      {error && !exams.length ? <ErrorBanner error={error} onRetry={() => reload()} retrying={fetching} title="Could not load your exams" /> : null}

      <section aria-labelledby="exams-heading" className="stack">
        <div className="row-between">
          <h2 id="exams-heading" className="section-title">
            Your exams
          </h2>
        </div>
        {showSkeleton ? (
          <div className="exam-grid">
            {[0, 1, 2].map((i) => (
              <ExamCardSkeleton key={i} />
            ))}
          </div>
        ) : exams.length === 0 && !error ? (
          <div className="card">
            <EmptyState
              illustration="exams"
              title="Create your first exam"
              description="Add the questions and a marking rubric, upload scanned answer sheets, and SARAS AI will grade them for your review."
              action={
                <Button variant="primary" iconLeft={<Plus size={16} aria-hidden="true" />} onClick={() => setCreating(true)}>
                  New exam
                </Button>
              }
            />
          </div>
        ) : (
          <div className="exam-grid">
            {exams.map((e) => (
              <ExamCard key={e.id} exam={e} />
            ))}
          </div>
        )}
      </section>

      <ExamFormModal
        open={creating}
        onClose={() => setCreating(false)}
        onSaved={(exam) => {
          upsert(exam);
          setCreating(false);
          navigate(`/exams/${exam.id}/rubric`);
        }}
      />
    </div>
  );
}
