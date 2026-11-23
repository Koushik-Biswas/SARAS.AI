// Step 4: Result. What the AI engines decided for every sheet: marks per question, the
// verdict and the AI's evaluation of each answer, before any teacher review.
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { BrainCircuit, ChevronDown, ChevronRight, CircleCheck, CircleMinus, CircleX, ClipboardCheck, Flag, Sparkles, Trophy, Users } from "lucide-react";
import * as api from "../api/endpoints";
import type { AnswerOut, ResultRow, ResultsOut } from "../api/types";
import {
  Badge,
  Banner,
  ButtonLink,
  Card,
  ConfidenceMeter,
  EmptyState,
  ErrorBanner,
  Segmented,
  Skeleton,
  SkeletonTable,
  StatTile,
  StatusPill,
  cx,
  type Tone,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { usePolling } from "../hooks/usePolling";
import { flagSpec, fmtMarks, INFO_FLAGS } from "../review/flags";
import { useExam } from "./ExamLayout";
import { StepFooter } from "./workflow";

const EVALUATED = new Set(["processed", "in_review", "approved"]);

type Filter = "all" | "check" | "clear";

const VERDICT: Record<string, { label: string; tone: Tone; icon: typeof CircleCheck }> = {
  correct: { label: "Correct", tone: "success", icon: CircleCheck },
  partially_correct: { label: "Partly correct", tone: "warning", icon: CircleMinus },
  incorrect: { label: "Incorrect", tone: "danger", icon: CircleX },
};

function verdictOf(a: AnswerOut): { label: string; tone: Tone; icon: typeof CircleCheck } {
  if (a.status === "blank") return { label: "Blank", tone: "neutral", icon: CircleMinus };
  if (a.verdict && VERDICT[a.verdict]) return VERDICT[a.verdict];
  const r = a.max_marks ? a.ai_marks / a.max_marks : 0;
  if (!a.grader) return { label: "Not marked", tone: "neutral", icon: CircleMinus };
  return r >= 0.85 ? VERDICT.correct : r > 0.15 ? VERDICT.partially_correct : VERDICT.incorrect;
}

const pctOf = (n: number | null | undefined, max: number) => (n === null || n === undefined || !max ? null : Math.round((n / max) * 100));

/** Score ring: AI total as a share of the maximum. */
function ScoreRing({ value, max, size = 52 }: { value: number | null; max: number; size?: number }) {
  const p = pctOf(value, max);
  const r = (size - 6) / 2;
  const c = 2 * Math.PI * r;
  const tone = p === null ? "none" : p >= 75 ? "high" : p >= 40 ? "mid" : "low";
  return (
    <span className={cx("ring", `ring--${tone}`)} style={{ width: size, height: size }} aria-hidden="true">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle className="ring__track" cx={size / 2} cy={size / 2} r={r} />
        <circle className="ring__value" cx={size / 2} cy={size / 2} r={r} strokeDasharray={c} strokeDashoffset={c * (1 - (p ?? 0) / 100)} />
      </svg>
      <span className="ring__text num">{p === null ? "-" : `${p}%`}</span>
    </span>
  );
}

/** One answer as the AI evaluated it. */
function AnswerEvaluation({ a }: { a: AnswerOut }) {
  const v = verdictOf(a);
  const VIcon = v.icon;
  const needsCheck = a.status === "needs_review" || a.status === "failed";
  const flags = a.flags.filter((f) => !INFO_FLAGS.has(f));
  const share = a.max_marks ? Math.max(0, Math.min(1, a.ai_marks / a.max_marks)) : 0;
  return (
    <li className={cx("eval", needsCheck && "eval--check")}>
      <div className="eval__head">
        <span className="eval__no num">Q{a.question_no}</span>
        <span className="eval__q">{a.question_prompt || <span className="subtle">Question {a.question_no}</span>}</span>
        <span className="eval__marks num">
          <strong>{fmtMarks(a.ai_marks)}</strong>
          <span className="subtle"> / {fmtMarks(a.max_marks)}</span>
        </span>
      </div>
      <div className="eval__bar" aria-hidden="true">
        <span className={`eval__fill eval__fill--${v.tone}`} style={{ width: `${share * 100}%` }} />
      </div>
      <div className="eval__grid">
        <div className="eval__block">
          <span className="section-label">Student's answer</span>
          <p className={cx("eval__answer", !a.ocr_text && "subtle")}>{a.ocr_text || (a.status === "blank" ? "Left blank." : "No text was read.")}</p>
        </div>
        <div className="eval__block eval__block--ai">
          <span className="section-label">
            <Sparkles size={12} aria-hidden="true" className="inline-icon" /> AI evaluation
          </span>
          <p className={cx("eval__feedback", !a.feedback && "subtle")}>
            {a.feedback || (a.grader ? "Marked without written feedback." : "The AI could not mark this answer; a teacher will set the marks in Review.")}
          </p>
        </div>
      </div>
      <div className="eval__meta">
        <Badge size="sm" tone={v.tone} icon={<VIcon size={12} aria-hidden="true" />}>
          {v.label}
        </Badge>
        {needsCheck ? (
          <Badge size="sm" tone="warning" icon={<Flag size={12} aria-hidden="true" />}>
            Needs a teacher check
          </Badge>
        ) : a.status === "reviewed" ? (
          <Badge size="sm" tone="info" icon={<ClipboardCheck size={12} aria-hidden="true" />}>
            Reviewed: {fmtMarks(a.final_marks)}
          </Badge>
        ) : null}
        {flags.slice(0, 3).map((f) => {
          const spec = flagSpec(f);
          return (
            <Badge key={f} size="sm" tone="neutral" title={spec.hint}>
              {spec.label}
            </Badge>
          );
        })}
        <span className="eval__conf">
          <ConfidenceMeter value={a.confidence} label={`AI confidence for Q${a.question_no}`} compact />
        </span>
      </div>
    </li>
  );
}

function SheetAnswers({ submissionId }: { submissionId: string }) {
  const q = useAsync(() => api.listAnswers(submissionId, { sort: "question" }), [submissionId]);
  if (q.loading && !q.data) {
    return (
      <div className="stack-sm sheet-eval__body">
        {[0, 1].map((i) => (
          <Skeleton key={i} height={96} radius={12} />
        ))}
      </div>
    );
  }
  if (q.error && !q.data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load the AI evaluation" />;
  return (
    <ol className="eval-list sheet-eval__body">
      {(q.data ?? []).map((a) => (
        <AnswerEvaluation key={a.answer_id} a={a} />
      ))}
    </ol>
  );
}

function SheetCard({ row, data, examId, open, onToggle }: { row: ResultRow; data: ResultsOut; examId: string; open: boolean; onToggle: () => void }) {
  const total = row.total_ai;
  const panelId = `sheet-${row.submission_id}`;
  return (
    <li className={cx("sheet-eval", open && "is-open")}>
      <div className="sheet-eval__head">
        <button type="button" className="sheet-eval__toggle" aria-expanded={open} aria-controls={panelId} onClick={onToggle}>
          {open ? <ChevronDown size={18} aria-hidden="true" /> : <ChevronRight size={18} aria-hidden="true" />}
          <ScoreRing value={total} max={data.max_total} />
          <span className="sheet-eval__who">
            <span className="cell-strong">{row.student_ref}</span>
            <span className="subtle num">
              AI marks {total ?? "-"} / {data.max_total}
              {row.total !== null && row.total !== total ? ` · after review ${row.total}` : ""}
            </span>
          </span>
        </button>
        <div className="sheet-eval__cells" aria-label={`Marks per question for ${row.student_ref}`}>
          {data.questions.map((qq, i) => {
            const c = row.cells[i];
            const p = c ? pctOf(c.ai_marks, c.max_marks) ?? 0 : null;
            return (
              <span
                key={qq.id}
                className={cx("qchip", p === null ? "qchip--none" : p >= 75 ? "qchip--high" : p >= 40 ? "qchip--mid" : "qchip--low", c && (c.status === "needs_review" || c.status === "failed") && "qchip--flag")}
                title={c ? `Q${qq.question_no}: ${c.ai_marks} / ${c.max_marks}${c.status === "needs_review" ? " (needs review)" : ""}` : `Q${qq.question_no}: no answer`}
              >
                <span className="qchip__no">Q{qq.question_no}</span>
                <span className="num">{c ? fmtMarks(c.ai_marks) : "-"}</span>
              </span>
            );
          })}
        </div>
        <div className="sheet-eval__side">
          <StatusPill kind="submission" status={row.status} size="sm" />
          {row.pending ? (
            <ButtonLink size="sm" variant="subtle" to={`/exams/${examId}/review/${row.submission_id}`} iconLeft={<Flag size={14} aria-hidden="true" />}>
              Review {row.pending}
            </ButtonLink>
          ) : null}
        </div>
      </div>
      {open ? (
        <div id={panelId}>
          <SheetAnswers submissionId={row.submission_id} />
        </div>
      ) : null}
    </li>
  );
}

export default function ResultsPage() {
  const { exam, reload: reloadExam } = useExam();
  const q = useAsync(() => api.getResults(exam.id), [exam.id]);
  const [filter, setFilter] = useState<Filter>("all");
  const [openIds, setOpenIds] = useState<Set<string>>(new Set());

  const data = q.data;
  const all = useMemo(() => data?.items ?? [], [data]);
  const inFlight = all.filter((r) => r.status === "uploaded" || r.status === "processing").length;
  usePolling(
    async () => {
      await q.reload(true);
      void reloadExam(true);
    },
    4000,
    inFlight > 0,
  );

  const rows = useMemo(() => all.filter((r) => EVALUATED.has(r.status)), [all]);
  const shown = rows.filter((r) => (filter === "check" ? r.pending > 0 : filter === "clear" ? r.pending === 0 : true));
  const totals = rows.map((r) => r.total_ai).filter((t): t is number => t !== null);
  const avg = totals.length ? Math.round((totals.reduce((a, b) => a + b, 0) / totals.length) * 100) / 100 : null;
  const best = totals.length ? Math.max(...totals) : null;
  const toCheck = rows.reduce((a, r) => a + r.pending, 0);

  const toggle = (id: string) =>
    setOpenIds((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  if (q.loading && !data) {
    return (
      <Card flush>
        <SkeletonTable rows={5} cols={5} />
      </Card>
    );
  }
  if (q.error && !data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load the AI results" />;

  if (!data || rows.length === 0) {
    return (
      <div className="stack">
        <Card>
          <EmptyState
            illustration="results"
            title={inFlight ? "The AI engines are still checking" : "No AI results yet"}
            description={
              inFlight
                ? `${inFlight} sheet${inFlight === 1 ? " is" : "s are"} being read and marked. Results appear here as each one finishes.`
                : "Upload answer sheets. The AI reads every answer, gives marks and explains its evaluation here."
            }
            action={
              <ButtonLink to={`/exams/${exam.id}/${inFlight ? "status" : "upload"}`} variant="primary" iconLeft={<BrainCircuit size={16} aria-hidden="true" />}>
                {inFlight ? "Watch the AI engines" : "Upload sheets"}
              </ButtonLink>
            }
          />
        </Card>
        <StepFooter exam={exam} slug="results" />
      </div>
    );
  }

  return (
    <div className="stack">
      {inFlight ? (
        <Banner tone="info" title={`AI still checking ${inFlight} sheet${inFlight === 1 ? "" : "s"}`}>
          Results below update automatically. <Link to={`/exams/${exam.id}/status`}>Watch the AI engines</Link>
        </Banner>
      ) : null}

      <div className="stat-grid">
        <StatTile label="Sheets evaluated" value={rows.length} icon={<Users size={16} />} sub={`${data.questions.length} question${data.questions.length === 1 ? "" : "s"} each`} />
        <StatTile label="Average AI marks" value={avg === null ? "-" : `${avg} / ${data.max_total}`} icon={<Sparkles size={16} />} tone="info" sub={avg === null ? undefined : `${pctOf(avg, data.max_total)}% of the total`} />
        <StatTile label="Highest" value={best === null ? "-" : `${best} / ${data.max_total}`} icon={<Trophy size={16} />} tone="success" />
        <StatTile label="Answers to check" value={toCheck} icon={<Flag size={16} />} tone={toCheck ? "warning" : "success"} sub={toCheck ? "Go to Review next" : "AI was confident on every answer"} />
      </div>

      <Card
        title="AI evaluation by student"
        subtitle="Open a sheet to see each answer, the marks the AI gave and why."
        actions={
          <Segmented<Filter>
            label="Filter sheets"
            size="sm"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: `All ${rows.length}` },
              { value: "check", label: `To check ${rows.filter((r) => r.pending > 0).length}` },
              { value: "clear", label: `Clear ${rows.filter((r) => r.pending === 0).length}` },
            ]}
          />
        }
      >
        {shown.length === 0 ? (
          <p className="subtle">No sheets match this filter.</p>
        ) : (
          <ul className="sheet-evals">
            {shown.map((r) => (
              <SheetCard key={r.submission_id} row={r} data={data} examId={exam.id} open={openIds.has(r.submission_id)} onToggle={() => toggle(r.submission_id)} />
            ))}
          </ul>
        )}
      </Card>

      <StepFooter
        exam={exam}
        slug="results"
        ready={inFlight === 0}
        hint={toCheck ? `${toCheck} answer${toCheck === 1 ? "" : "s"} flagged for a manual check` : "Nothing flagged: a quick look in Review is enough"}
      />
    </div>
  );
}
