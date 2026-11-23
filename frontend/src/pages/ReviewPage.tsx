// Review tab (PRD 17.1 screen 6).
// - /exams/:examId/review                 -> sheet picker: every processed sheet with its pending count
// - /exams/:examId/review/:submissionId   -> review workspace for one student sheet
//   (?answer=<answer_id> deep-links to an answer)
import "../review/review.css";
import { ReviewWorkspace } from "../review/ReviewWorkspace";
import { ReviewQueue } from "./ReviewQueue";

export interface ReviewPageProps {
  examId: string;
  /** Absent on /exams/:examId/review (queue landing). */
  submissionId?: string;
}

export default function ReviewPage({ examId, submissionId }: ReviewPageProps) {
  if (!submissionId) return <ReviewQueue examId={examId} />;
  // Keyed so every piece of per-sheet state (drafts, filters, touched answers) resets on switch.
  return <ReviewWorkspace key={submissionId} examId={examId} submissionId={submissionId} />;
}
