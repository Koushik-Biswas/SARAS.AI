// The exam workflow: 1 Set rubric -> 2 Upload -> 3 AI status -> 4 Result -> 5 Review ->
// 6 Final approval. One definition drives the stepper in the exam header, the
// "continue" bar under each step and the exam's landing redirect.
import { NavLink, useLocation } from "react-router-dom";
import { ArrowLeft, ArrowRight, BrainCircuit, ChartColumn, Check, ClipboardCheck, CloudUpload, ListChecks, ShieldCheck, Sparkles, type LucideIcon } from "lucide-react";
import type { ExamOut, ExamStats } from "../api/types";
import { ButtonLink, cx } from "../components";

export type StepSlug = "rubric" | "upload" | "status" | "results" | "review" | "approval";
export type StepState = "done" | "current" | "todo" | "attention" | "active";

interface StepDef {
  slug: StepSlug;
  label: string;
  /** Short label for the continue bar ("Continue to ..."). */
  action: string;
  icon: LucideIcon;
}

export const WORKFLOW: StepDef[] = [
  { slug: "rubric", label: "Set rubric", action: "Set rubric", icon: ListChecks },
  { slug: "upload", label: "Upload", action: "Upload answer sheets", icon: CloudUpload },
  { slug: "status", label: "AI status", action: "Watch the AI engines", icon: BrainCircuit },
  { slug: "results", label: "Result", action: "See AI results", icon: Sparkles },
  { slug: "review", label: "Review", action: "Review manually", icon: ClipboardCheck },
  { slug: "approval", label: "Final approval", action: "Final approval", icon: ShieldCheck },
];

export const ANALYTICS = { slug: "analytics", label: "Analytics", icon: ChartColumn } as const;

/** Sheets whose AI evaluation is finished (processed, in review or approved). */
const evaluated = (s: ExamStats) => s.processed;

/** Whether each step is complete, given the exam's numbers. */
export function stepDone(slug: StepSlug, s: ExamStats): boolean {
  switch (slug) {
    case "rubric":
      return s.questions > 0;
    case "upload":
      return s.sheets > 0;
    case "status":
      return s.sheets > 0 && s.processing === 0 && evaluated(s) > 0;
    case "results":
      return evaluated(s) > 0 && s.processing === 0;
    case "review":
      return evaluated(s) > 0 && s.needs_review === 0;
    case "approval":
      return evaluated(s) > 0 && s.approved === evaluated(s);
  }
}

/** The first step that still needs the teacher (or the AI). */
export function nextStep(s: ExamStats): StepSlug {
  if (s.questions === 0) return "rubric";
  if (s.sheets === 0) return "upload";
  if (s.processing > 0 || evaluated(s) === 0) return "status";
  if (s.approved > 0 && s.approved === evaluated(s)) return "approval";
  return "results"; // AI marks are ready: Result, then Review, then Final approval
}

function stepCaption(slug: StepSlug, s: ExamStats): string {
  const n = (k: number, one: string, many = `${one}s`) => `${k} ${k === 1 ? one : many}`;
  switch (slug) {
    case "rubric":
      return s.questions ? n(s.questions, "question") : "Questions & marks";
    case "upload":
      return s.sheets ? n(s.sheets, "sheet") : "Answer sheets";
    case "status":
      if (s.processing) return `AI checking ${s.processing}`;
      if (s.failed) return `${s.failed} failed`;
      return evaluated(s) ? "Engines idle" : "AI engines";
    case "results":
      return evaluated(s) ? `${n(evaluated(s), "sheet")} marked` : "AI evaluation";
    case "review":
      return s.needs_review ? `${n(s.needs_review, "answer")} to check` : evaluated(s) ? "All checked" : "Manual check";
    case "approval":
      return evaluated(s) ? `${s.approved}/${evaluated(s)} approved` : "Lock & export";
  }
}

function stepState(slug: StepSlug, s: ExamStats, current: boolean): StepState {
  if (current) return "current";
  if (slug === "status" && s.processing > 0) return "active";
  if ((slug === "status" && s.failed > 0) || (slug === "review" && s.needs_review > 0)) return "attention";
  return stepDone(slug, s) ? "done" : "todo";
}

/** The workflow stepper shown under the exam header (navigation between the 6 steps). */
export function WorkflowStepper({ exam }: { exam: ExamOut }) {
  const { pathname } = useLocation();
  const base = `/exams/${exam.id}`;
  const currentSlug =
    WORKFLOW.find((w) => pathname === `${base}/${w.slug}` || pathname.startsWith(`${base}/${w.slug}/`))?.slug ??
    (pathname.startsWith(`${base}/regions/`) ? "review" : undefined);
  const s = exam.stats;
  const analyticsActive = pathname.startsWith(`${base}/${ANALYTICS.slug}`);
  const AnalyticsIcon = ANALYTICS.icon;

  return (
    <nav className="wf" aria-label="Exam workflow">
      <ol className="wf__list">
        {WORKFLOW.map((w, i) => {
          const state = stepState(w.slug, s, w.slug === currentSlug);
          const done = stepDone(w.slug, s);
          const Icon = w.icon;
          return (
            <li key={w.slug} className={cx("wf__item", `is-${state}`, done && "is-complete")}>
              <NavLink
                to={`${base}/${w.slug}`}
                className="wf__step"
                aria-current={state === "current" ? "step" : undefined}
                aria-label={`Step ${i + 1}: ${w.label}, ${stepCaption(w.slug, s)}${done ? ", complete" : ""}`}
              >
                <span className="wf__dot" aria-hidden="true">
                  {done && state !== "current" ? <Check size={15} strokeWidth={3} /> : <Icon size={15} aria-hidden="true" />}
                  <span className="wf__num">{i + 1}</span>
                </span>
                <span className="wf__text">
                  <span className="wf__label">{w.label}</span>
                  <span className="wf__caption">{stepCaption(w.slug, s)}</span>
                </span>
              </NavLink>
              {i < WORKFLOW.length - 1 ? <span className={cx("wf__bar", done && "is-filled")} aria-hidden="true" /> : null}
            </li>
          );
        })}
      </ol>
      <NavLink to={`${base}/${ANALYTICS.slug}`} className={cx("wf__extra", analyticsActive && "is-current")} aria-current={analyticsActive ? "page" : undefined}>
        <AnalyticsIcon size={15} aria-hidden="true" />
        <span>{ANALYTICS.label}</span>
      </NavLink>
    </nav>
  );
}

/** "Back / Continue" bar at the bottom of a step. `ready` highlights the continue button. */
export function StepFooter({ exam, slug, ready, hint }: { exam: ExamOut; slug: StepSlug; ready?: boolean; hint?: string }) {
  const i = WORKFLOW.findIndex((w) => w.slug === slug);
  const prev = WORKFLOW[i - 1];
  const next = WORKFLOW[i + 1];
  const isReady = ready ?? stepDone(slug, exam.stats);
  return (
    <div className="wf-footer" role="navigation" aria-label="Workflow steps">
      {prev ? (
        <ButtonLink to={`/exams/${exam.id}/${prev.slug}`} variant="ghost" iconLeft={<ArrowLeft size={16} aria-hidden="true" />}>
          {prev.label}
        </ButtonLink>
      ) : (
        <span />
      )}
      <div className="wf-footer__right">
        {hint ? <span className="wf-footer__hint">{hint}</span> : null}
        {next ? (
          <ButtonLink to={`/exams/${exam.id}/${next.slug}`} variant={isReady ? "primary" : "secondary"} iconRight={<ArrowRight size={16} aria-hidden="true" />}>
            Step {i + 2}: {next.action}
          </ButtonLink>
        ) : null}
      </div>
    </div>
  );
}
