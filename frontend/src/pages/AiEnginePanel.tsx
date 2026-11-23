// Step 3 hero: the AI engines at work. Each uploaded sheet moves through four engines
// (read the sheet -> split answers -> evaluate each answer -> generate marks); this panel
// shows which engines are switched on, what each is doing right now and overall progress.
// Everything comes from the worker's progress reports (SubmissionOut.progress): the
// engine a sheet is in is the pipeline stage it reported, and the percentage counts
// finished stages, pages read and questions marked - never a timer.
import { ArrowRight, BrainCircuit, Calculator, CircleCheck, ScanText, Scissors, Sparkles } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ExamOut, SubmissionOut } from "../api/types";
import { ButtonLink, cx } from "../components";
import { currentStage, sheetProgress } from "./pipeline";

type EngineKey = "read" | "split" | "evaluate" | "marks";

interface Engine {
  key: EngineKey;
  title: string;
  detail: string;
  icon: LucideIcon;
}

const ENGINES: Engine[] = [
  { key: "read", title: "Reading the sheet", detail: "Preprocessing and handwriting recognition", icon: ScanText },
  { key: "split", title: "Splitting answers", detail: "Question identification and answer extraction", icon: Scissors },
  { key: "evaluate", title: "Evaluating answers", detail: "Rubric interpretation and semantic analysis", icon: Sparkles },
  { key: "marks", title: "Generating marks", detail: "Mark allocation, verification and results", icon: Calculator },
];

/** Pipeline stage (worker progress) -> engine card. */
const STAGE_ENGINE: Record<string, EngineKey> = {
  preprocess: "read",
  recognition: "read",
  identification: "split",
  extraction: "split",
  rubric: "evaluate",
  analysis: "evaluate",
  marking: "marks",
  verification: "marks",
  results: "marks",
};

const DONE = new Set(["processed", "in_review", "approved"]);
/** Position in the pipeline: a sheet has passed engine i when its rank is above i. */
const RANK: Record<string, number> = { failed: -1, queued: -1, read: 0, split: 1, evaluate: 2, marks: 3, done: 4 };

/** Which engine a sheet is in (null = waiting in the queue, "done" = finished). */
function engineOf(s: SubmissionOut): EngineKey | "queued" | "done" | "failed" {
  if (s.status === "failed") return "failed";
  if (DONE.has(s.status) || s.stage === "done") return "done";
  if (s.status === "uploaded") return "queued";
  const cur = currentStage(s);
  if (cur) return STAGE_ENGINE[cur.key] ?? "queued";
  if (s.stage === "ocr") return "read";
  if (s.stage === "segmentation") return "split";
  if (s.stage === "grading") return "evaluate";
  return "queued";
}

export function AiEnginePanel({ exam, subs }: { exam: ExamOut; subs: SubmissionOut[] }) {
  const by = (k: string) => subs.filter((s) => engineOf(s) === k);
  const queued = by("queued").length;
  const done = by("done").length;
  const failed = by("failed").length;
  const active = subs.length - done - failed;
  const overall = subs.length ? subs.reduce((a, s) => a + sheetProgress(s), 0) / subs.length : 0;
  const pct = Math.round(overall * 100);
  const state: "on" | "idle" | "complete" = active > 0 ? "on" : done > 0 ? "complete" : "idle";
  const working: Record<EngineKey, SubmissionOut[]> = { read: by("read"), split: by("split"), evaluate: by("evaluate"), marks: by("marks") };
  // What each active sheet's worker last reported: its stage and that stage's own detail.
  const ticker = subs
    .filter((s) => ["read", "split", "evaluate", "marks"].includes(engineOf(s)))
    .slice(0, 4)
    .map((s) => {
      const cur = currentStage(s);
      const what = cur ? `${cur.label}${cur.detail ? `: ${cur.detail}` : ""}` : `page ${s.pages_done} of ${s.page_count || "?"} done`;
      return { id: s.id, text: `${s.student_ref}: ${what}` };
    });

  const headline =
    state === "on"
      ? `AI engines on: checking ${active} sheet${active === 1 ? "" : "s"}`
      : state === "complete"
        ? `AI evaluation complete: ${done} sheet${done === 1 ? "" : "s"} marked`
        : "AI engines on standby";
  const sub =
    state === "on"
      ? `${done} of ${subs.length} finished${queued ? ` · ${queued} waiting in queue` : ""}${failed ? ` · ${failed} failed` : ""}. Updates live.`
      : state === "complete"
        ? `Every answer has AI marks and an evaluation.${failed ? ` ${failed} sheet${failed === 1 ? "" : "s"} failed: retry below.` : " Continue to the results."}`
        : "Upload answer sheets to switch the engines on.";

  return (
    <section className={cx("engine", `engine--${state}`)} aria-labelledby="engine-title">
      <div className="engine__top">
        <div className="brain" aria-hidden="true">
          <span className="brain__halo" />
          <span className="brain__ring brain__ring--1" />
          <span className="brain__ring brain__ring--2" />
          <span className="brain__orbit">
            <span />
            <span />
            <span />
          </span>
          <span className="brain__core">{state === "complete" ? <CircleCheck size={30} /> : <BrainCircuit size={30} />}</span>
        </div>
        <div className="engine__head">
          <span className={cx("engine__status", `is-${state}`)}>
            <span className="engine__led" aria-hidden="true" />
            {state === "on" ? "Engines active" : state === "complete" ? "Evaluation complete" : "Standby"}
          </span>
          <h2 id="engine-title" className="engine__title">
            {headline}
          </h2>
          <p className="engine__sub">{sub}</p>
          <div className="engine__progress" role="progressbar" aria-label="Overall AI progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
            <span className="engine__progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
        <div className="engine__pct">
          <span className="engine__pct-value num">{pct}%</span>
          {state === "complete" ? (
            <ButtonLink to={`/exams/${exam.id}/results`} variant="primary" size="sm" iconRight={<ArrowRight size={14} aria-hidden="true" />}>
              See results
            </ButtonLink>
          ) : null}
        </div>
      </div>

      <ol className="engine__pipeline">
        {ENGINES.map((e, i) => {
          const Icon = e.icon;
          const now = working[e.key].length;
          const isOn = now > 0;
          const lit = subs.some((s) => RANK[engineOf(s)] > i); // some sheet is already past this engine
          return (
            <li key={e.key} className={cx("engine__step", isOn && "is-on", lit && !isOn && "is-lit")}>
              <span className="engine__step-icon" aria-hidden="true">
                <Icon size={18} />
              </span>
              <span className="engine__step-text">
                <span className="engine__step-title">
                  <span className="engine__step-no num">{i + 1}</span> {e.title}
                </span>
                <span className="engine__step-detail">{e.detail}</span>
                <span className="engine__step-count">
                  {isOn ? `${now} sheet${now === 1 ? "" : "s"} now` : lit ? "Done" : "Waiting"}
                </span>
              </span>
              {isOn ? <span className="engine__scan" aria-hidden="true" /> : null}
            </li>
          );
        })}
      </ol>

      {ticker.length ? (
        <ul className="engine__ticker" aria-live="polite" aria-label="What the AI is doing now">
          {ticker.map((t) => (
            <li key={t.id}>
              <span className="engine__tick-dot" aria-hidden="true" />
              {t.text}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
