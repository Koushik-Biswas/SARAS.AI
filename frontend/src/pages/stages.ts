// Human-readable processing stages. The worker reports: queued -> ocr (render, clean, read)
// -> segmentation -> grading -> done; "retrying" while waiting to retry after an error. A
// failed sheet keeps the step it stopped at ("failed" only when no step is known).
import type { SubmissionOut } from "../api/types";

const STAGES: Record<string, string> = {
  queued: "Waiting in queue",
  retrying: "Retrying after an error",
  ocr: "AI reading the answer sheet",
  segmentation: "Splitting answers",
  grading: "AI generating marks",
  done: "Done",
  failed: "Failed",
};

/** Steps a failure can be reported at, phrased to follow "Failed while ...". */
const FAILED_AT: Record<string, string> = {
  ocr: "reading the answer sheet",
  segmentation: "splitting answers",
  grading: "generating marks",
};

export function stageLabel(s: Pick<SubmissionOut, "status" | "stage">): string {
  if (s.status === "failed") return s.stage && FAILED_AT[s.stage] ? `Failed while ${FAILED_AT[s.stage]}` : "Failed";
  if (s.status === "uploaded") return s.stage === "retrying" ? STAGES.retrying : STAGES.queued;
  if (!s.stage) return "-";
  return STAGES[s.stage] ?? s.stage.replace(/_/g, " ");
}
