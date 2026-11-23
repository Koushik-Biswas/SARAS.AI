// Real pipeline progress of a sheet, computed only from what the worker reported
// (SubmissionOut.progress): finished stages, and inside the stages that can measure
// it, the pages read / questions marked so far. Nothing here is animated or guessed.
import type { PipelineProgress, PipelineStageKey, SubmissionOut } from "../api/types";

/** Share of a sheet's processing each stage represents (sums to 1). */
const WEIGHT: Record<PipelineStageKey, number> = {
  preprocess: 0.08,
  recognition: 0.37,
  identification: 0.04,
  extraction: 0.03,
  rubric: 0.04,
  analysis: 0.26,
  marking: 0.06,
  verification: 0.09,
  results: 0.03,
};

const FINISHED = new Set(["done", "warning", "failed", "skipped"]);
const DONE_STATUSES = new Set(["processed", "in_review", "approved"]);

/** 0..1 within a running stage, from real counts (0 when the stage cannot measure it). */
function within(p: PipelineProgress, key: PipelineStageKey): number {
  const c = p.counts;
  const writtenPages = Math.max(0, c.pages_total - c.pages_blank);
  if (key === "preprocess" && c.pages_total) return Math.min(1, c.pages_preprocessed / c.pages_total);
  if (key === "recognition" && writtenPages) return Math.min(1, (c.pages_read + c.pages_text_layer + c.pages_unread) / writtenPages);
  if (key === "marking" && c.questions_total) return Math.min(1, c.questions_marked / c.questions_total);
  return 0;
}

/** 0..1 progress of one sheet, from backend events only. */
export function sheetProgress(s: SubmissionOut): number {
  if (DONE_STATUSES.has(s.status)) return 1;
  const p = s.progress;
  if (s.status === "failed") return p ? stagesFraction(p) : 0;
  if (!p || s.status === "uploaded") return 0;
  return stagesFraction(p);
}

function stagesFraction(p: PipelineProgress): number {
  let total = 0;
  for (const st of p.stages) {
    const w = WEIGHT[st.key] ?? 0;
    if (FINISHED.has(st.state)) total += w;
    else if (st.state === "running") total += w * within(p, st.key);
  }
  return Math.min(1, total);
}

/** The stage label a sheet is in right now, from its progress (null if unknown). */
export function currentStage(s: SubmissionOut): { key: PipelineStageKey; label: string; detail: string } | null {
  const p = s.progress;
  if (!p || p.finished || !p.current) return null;
  const st = p.stages.find((x) => x.key === p.current);
  return st ? { key: st.key, label: st.label, detail: st.detail } : null;
}

export function formatSeconds(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) return "-";
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  const m = Math.floor(sec / 60);
  return `${m} min ${Math.round(sec - m * 60)} s`;
}
