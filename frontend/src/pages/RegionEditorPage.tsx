// Region editor (PRD 17.1 screen 7): /exams/:examId/regions/:submissionId (?page=<n>).
import "../review/review.css";
import { RegionEditor } from "../review/RegionEditor";

export interface RegionEditorPageProps {
  examId: string;
  submissionId: string;
}

export default function RegionEditorPage({ examId, submissionId }: RegionEditorPageProps) {
  return <RegionEditor key={submissionId} examId={examId} submissionId={submissionId} />;
}
