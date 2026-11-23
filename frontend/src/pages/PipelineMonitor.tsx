// Step 3: the real-time processing monitor. One sheet at a time (the one being
// processed, else the most recent), its nine pipeline stages exactly as the worker
// reported them - state, measured duration and what happened - plus real counts
// (pages read, questions marked, uncertain answers, failures, AI requests and tokens).
// Nothing is simulated: every value comes from SubmissionOut.progress.
import { useEffect, useMemo, useState } from "react";
import { CircleAlert, CircleCheck, CircleDashed, CircleX, LoaderCircle, MinusCircle } from "lucide-react";
import type { PipelineStageState, SubmissionOut } from "../api/types";
import { Badge, Card, ProgressBar, Select, cx } from "../components";
import { formatSeconds, sheetProgress } from "./pipeline";

const STATE_ICON: Record<PipelineStageState, JSX.Element> = {
  pending: <CircleDashed size={16} aria-hidden="true" />,
  running: <LoaderCircle size={16} aria-hidden="true" className="pm__spin" />,
  done: <CircleCheck size={16} aria-hidden="true" />,
  warning: <CircleAlert size={16} aria-hidden="true" />,
  failed: <CircleX size={16} aria-hidden="true" />,
  skipped: <MinusCircle size={16} aria-hidden="true" />,
};
const STATE_TEXT: Record<PipelineStageState, string> = {
  pending: "Waiting",
  running: "Running",
  done: "Done",
  warning: "Done with warnings",
  failed: "Failed",
  skipped: "Skipped",
};

/** The sheet to show by default: processing first, then the most recently created. */
function defaultSheet(subs: SubmissionOut[]): SubmissionOut | undefined {
  const active = subs.find((s) => s.status === "processing");
  if (active) return active;
  return [...subs].filter((s) => s.progress).sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
}

export function PipelineMonitor({ subs }: { subs: SubmissionOut[] }) {
  const withProgress = useMemo(() => subs.filter((s) => s.progress || s.status === "processing"), [subs]);
  const [picked, setPicked] = useState<string | null>(null);
  const sheet = (picked && withProgress.find((s) => s.id === picked)) || defaultSheet(withProgress);
  const p = sheet?.progress ?? null;

  // Elapsed time between polls: the worker's last reported elapsed time plus the time since
  // this poll arrived (real wall clock), only while the sheet is still running.
  const [fetchedAt, setFetchedAt] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => setFetchedAt(Date.now()), [p?.updated_at, p?.elapsed_seconds]);
  const running = !!p && !p.finished && sheet?.status === "processing";
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [running]);

  if (!sheet || !p) {
    return (
      <Card title="Live processing monitor" subtitle="Stages appear here as soon as the worker starts a sheet.">
        <p className="muted">No sheet has started processing yet.</p>
      </Card>
    );
  }

  const elapsed = running ? p.elapsed_seconds + Math.max(0, (now - fetchedAt) / 1000) : p.elapsed_seconds;
  const c = p.counts;
  const failed = sheet.status === "failed";
  const pct = sheetProgress(sheet);
  const writtenPages = Math.max(0, c.pages_total - c.pages_blank);
  const stats: { label: string; value: string; tone?: "warning" | "danger" | "success" }[] = [
    { label: "Pages read", value: `${c.pages_read + c.pages_text_layer} / ${writtenPages}`, tone: c.pages_unread ? "danger" : undefined },
    { label: "Answers extracted", value: `${c.answers_extracted} / ${c.questions_total}` },
    { label: "Questions marked", value: `${c.questions_marked} / ${c.questions_total}`, tone: p.finished ? "success" : undefined },
    { label: "Uncertain answers", value: String(c.uncertain), tone: c.uncertain ? "warning" : undefined },
    { label: "Re-marked (verification)", value: `${c.reevaluated} / ${c.verification_candidates}` },
    { label: "Failed operations", value: String(c.failed_operations + c.pages_unread), tone: c.failed_operations + c.pages_unread ? "danger" : undefined },
    { label: "AI requests", value: c.ai_requests || c.ai_cache_hits ? `${c.ai_requests}${c.ai_cache_hits ? ` + ${c.ai_cache_hits} cached` : ""}` : "-" },
    { label: "Tokens", value: c.tokens_in + c.tokens_out ? (c.tokens_in + c.tokens_out).toLocaleString() : "-" },
  ];

  return (
    <Card
      title="Live processing monitor"
      subtitle={
        p.finished
          ? `Finished in ${formatSeconds(p.elapsed_seconds)}. Marks are provisional until a teacher approves them.`
          : failed
            ? "Processing stopped. The stage it stopped at is marked below."
            : "Every step below is reported by the evaluation engine as it happens."
      }
      actions={
        withProgress.length > 1 ? (
          <Select
            aria-label="Sheet to monitor"
            value={sheet.id}
            onChange={(e) => setPicked(e.target.value)}
            className="pm__select"
          >
            {withProgress.map((s) => (
              <option key={s.id} value={s.id}>
                {s.student_ref}
                {s.status === "processing" ? " (processing)" : ""}
              </option>
            ))}
          </Select>
        ) : undefined
      }
    >
      <div className="pm">
        <div className="pm__head">
          <div className="pm__who">
            <span className="cell-strong">{sheet.student_ref}</span>
            <Badge tone={p.finished ? "success" : failed ? "danger" : "accent"} size="sm">
              {p.finished ? "Evaluation complete" : failed ? "Failed" : sheet.status === "processing" ? "Processing" : "Queued"}
            </Badge>
          </div>
          <span className="pm__elapsed num" aria-label="Processing time">
            {formatSeconds(elapsed)}
          </span>
        </div>
        <ProgressBar
          label={`${sheet.student_ref}: evaluation progress`}
          value={pct}
          tone={failed ? "danger" : p.finished ? "success" : "accent"}
          showLabel
          valueText={`${Math.round(pct * 100)}%`}
        />

        <ol className="pm__stages">
          {p.stages.map((st, i) => {
            const state: PipelineStageState = failed && st.state === "running" ? "failed" : st.state;
            return (
              <li key={st.key} className={cx("pm__stage", `is-${state}`)} aria-current={state === "running" ? "step" : undefined}>
                <span className="pm__icon" title={STATE_TEXT[state]}>
                  {STATE_ICON[state]}
                </span>
                <span className="pm__text">
                  <span className="pm__label">
                    <span className="pm__no num">{i + 1}</span> {st.label}
                    <span className="sr-only">: {STATE_TEXT[state]}</span>
                  </span>
                  {st.detail ? <span className="pm__detail">{st.detail}</span> : null}
                </span>
                <span className="pm__time num">{st.seconds !== null ? formatSeconds(st.seconds) : state === "running" ? "…" : ""}</span>
              </li>
            );
          })}
        </ol>

        <dl className="pm__stats">
          {stats.map((s) => (
            <div key={s.label} className={cx("pm__stat", s.tone && `is-${s.tone}`)}>
              <dt>{s.label}</dt>
              <dd className="num">{s.value}</dd>
            </div>
          ))}
        </dl>

        {p.events.length ? (
          <ul className="pm__events" aria-label="Notable events">
            {p.events
              .slice()
              .reverse()
              .map((e, i) => (
                <li key={i} className={cx("pm__event", `is-${e.level}`)}>
                  <span className="num">{formatSeconds(e.at)}</span> {e.text}
                </li>
              ))}
          </ul>
        ) : null}
      </div>
    </Card>
  );
}
