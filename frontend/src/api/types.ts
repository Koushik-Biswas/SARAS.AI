// TypeScript mirror of backend/app/schemas.py and the /api/v1 contract.
// Timestamps are UTC ISO-8601 strings; dates are "YYYY-MM-DD".

export type Role = "teacher" | "admin" | string;
export type QuestionType = "short" | "long" | "paragraph" | "maths";

export type SubmissionStatus =
  | "uploaded"
  | "processing"
  | "processed"
  | "in_review"
  | "approved"
  | "failed";

export type AnswerStatus = "needs_review" | "auto_accepted" | "reviewed" | "blank" | "failed";
export type AnswerFilter = "all" | AnswerStatus;
export type AnswerSort = "priority" | "question";
export type KeyPointStatus = "matched" | "partial" | "missed" | "contradicted";

// ---------- errors ----------
export interface FieldErrorDetail {
  field: string;
  message: string;
}

export interface ErrorBody {
  error: { code: string; message: string; details?: unknown };
}

// ---------- pagination ----------
export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface PageParams {
  page?: number;
  page_size?: number;
}

// ---------- auth ----------
export interface UserOut {
  id: string;
  name: string;
  email: string;
  role: Role;
}

export interface TokenOut {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: UserOut;
}

export interface RegisterIn {
  name: string;
  email: string;
  password: string;
}

export interface LoginIn {
  email: string;
  password: string;
}

// ---------- exams ----------
export interface ExamIn {
  title: string;
  subject?: string | null;
  date?: string | null;
  total_marks: number;
}

export type ExamPatch = Partial<ExamIn>;

export interface ExamStats {
  questions: number;
  sheets: number;
  processed: number;
  processing: number;
  failed: number;
  approved: number;
  needs_review: number;
  average_marks: number | null;
}

export interface ExamOut {
  id: string;
  title: string;
  subject: string | null;
  date: string | null;
  total_marks: number;
  created_at: string;
  rubric_marks: number;
  stats: ExamStats;
  /** Only the exam's owner may change it (rubric, sheets, marks); admins read others' exams. */
  can_edit?: boolean;
}

export interface KeyPointIn {
  text: string;
  weight: number;
  keywords: string[];
}

export interface QuestionIn {
  question_no: number;
  type: QuestionType;
  max_marks: number;
  prompt?: string | null;
  model_answer?: string | null;
  key_points: KeyPointIn[];
}

export interface QuestionsIn {
  questions: QuestionIn[];
}

export interface StoredKeyPoint {
  text: string;
  weight: number;
  keywords?: string[];
}

export interface QuestionOut {
  id: string;
  question_no: number;
  type: QuestionType;
  max_marks: number;
  prompt: string | null;
  model_answer: string | null;
  key_points: StoredKeyPoint[];
}

export interface RegradeOut {
  regraded: number;
}

// ---------- results / analytics ----------
/** POST /exams/{id}/questions/extract: a DRAFT rubric read from the question paper (not saved). */
export interface PaperQuestion {
  question_no: number;
  type: QuestionType;
  /** null when no marks were printed for the question. */
  max_marks: number | null;
  prompt: string;
}

export interface QuestionPaperOut {
  questions: PaperQuestion[];
  /** "ai:gemini" / "ai:groq" (AI reader) or "text" (rule-based reader of the PDF text). */
  source: string;
  pages: number;
  warnings: string[];
}

export interface ResultCell {
  answer_id: string;
  marks: number;
  ai_marks: number;
  status: AnswerStatus;
  max_marks: number;
}

export interface ResultRow {
  submission_id: string;
  student_ref: string;
  status: SubmissionStatus;
  cells: (ResultCell | null)[];
  total: number | null;
  total_ai: number | null;
  pending: number;
}

export interface ResultsOut {
  questions: { id: string; question_no: number; max_marks: number }[];
  max_total: number;
  items: ResultRow[];
}

export interface DistributionBucket {
  from_pct: number;
  to_pct: number;
  count: number;
}

export interface QuestionAnalytics {
  question_no: number;
  type: QuestionType;
  max_marks: number;
  average: number | null;
  average_pct: number | null;
  blank: number;
  most_missed: [string, number][];
}

export interface AnalyticsOut {
  sheets: number;
  max_total: number;
  average: number | null;
  median: number | null;
  stdev: number | null;
  highest: number | null;
  lowest: number | null;
  distribution: DistributionBucket[];
  questions: QuestionAnalytics[];
  hardest: number[];
  auto_accepted_share: number | null;
  mean_confidence: number | null;
  override_rate: number | null;
  mean_abs_override: number | null;
  flags: [string, number][];
}

// ---------- submissions ----------
export interface SubmissionOut {
  id: string;
  exam_id: string;
  student_ref: string;
  original_filename: string | null;
  status: SubmissionStatus;
  stage: string | null;
  page_count: number;
  pages_done: number;
  error: string | null;
  total_ai_marks: number | null;
  total_final_marks: number | null;
  ocr_engine: string | null;
  grader: string | null;
  flags: string[];
  page_flags: Record<string, string[]>;
  timings: Record<string, number>;
  /** Live pipeline progress written by the worker at every real event (null before processing starts). */
  progress?: PipelineProgress | null;
  created_at: string;
  approved_at: string | null;
  answers_needing_review: number;
}

export type PipelineStageKey =
  | "preprocess"
  | "recognition"
  | "identification"
  | "extraction"
  | "rubric"
  | "analysis"
  | "marking"
  | "verification"
  | "results";
export type PipelineStageState = "pending" | "running" | "done" | "warning" | "failed" | "skipped";

export interface PipelineStage {
  key: PipelineStageKey;
  label: string;
  state: PipelineStageState;
  /** Seconds the stage actually ran (null until it has run). */
  seconds: number | null;
  detail: string;
}

export interface PipelineCounts {
  pages_total: number;
  pages_preprocessed: number;
  pages_blank: number;
  pages_text_layer: number;
  pages_read: number;
  pages_unread: number;
  questions_total: number;
  answers_extracted: number;
  answers_blank: number;
  questions_marked: number;
  uncertain: number;
  verification_candidates: number;
  reevaluated: number;
  failed_operations: number;
  ai_requests: number;
  ai_cache_hits: number;
  ai_failed_requests: number;
  tokens_in: number;
  tokens_out: number;
}

export interface PipelineProgress {
  version: number;
  started_at: string;
  updated_at: string;
  elapsed_seconds: number;
  current: PipelineStageKey | null;
  finished: boolean;
  stages: PipelineStage[];
  counts: PipelineCounts;
  events: { at: number; level: "info" | "warning" | "error"; text: string }[];
}

export interface StatusOut {
  id: string;
  status: SubmissionStatus;
  stage: string | null;
  pages_done: number;
  pages_total: number;
  errors: string[];
  attempts: number;
  progress?: PipelineProgress | null;
}

export interface UploadRejection {
  filename: string;
  student_ref: string | null;
  code: string;
  message: string;
}

export interface UploadOut {
  submissions: SubmissionOut[];
  rejected: UploadRejection[];
}

export interface UploadProgress {
  loaded: number;
  total: number;
  fraction: number;
}

export interface ApproveAllOut {
  approved: string[];
  blocked: { submission_id: string; student_ref: string; pending: number[] }[];
}

// ---------- answers / review ----------
export interface KeyPointOut {
  index: number;
  text: string;
  weight: number;
  matched: boolean;
  status: KeyPointStatus;
  awarded: number;
  entailment_prob: number;
  contradiction_prob: number;
  evidence_sentence: string | null;
  /** Additive (grading ensemble): reason code or the AI's short reason, deciding grader, dispute, every grader's status. */
  reason?: string;
  source?: string | null;
  disputed?: boolean;
  statuses_by_grader?: Record<string, string>;
}

export interface AnswerOut {
  answer_id: string;
  submission_id: string;
  question_id: string;
  question_no: number;
  question_type: QuestionType;
  question_prompt: string | null;
  status: AnswerStatus;
  ocr_text: string;
  text_corrected: boolean;
  ocr_confidence: number;
  ai_marks: number;
  final_marks: number | null;
  max_marks: number;
  confidence: number;
  grading_confidence: number;
  segmentation_confidence: number;
  grader: string | null;
  key_points: KeyPointOut[];
  matched_points: string[];
  missed_points: string[];
  flags: string[];
  comment: string | null;
  region_ids: string[];
  locked: boolean;
  /** Additive (grading ensemble): verdict, feedback and the decision trace (codes and scores only). */
  verdict?: "correct" | "partially_correct" | "incorrect" | "ungraded" | string | null;
  feedback?: string | null;
  grading_trace?: Record<string, unknown> | null;
}

export interface ReviewIn {
  final_marks?: number | null;
  corrected_text?: string | null;
  comment?: string | null;
  reason?: string | null;
  approve?: boolean;
}

export interface FlagIn {
  reason?: string | null;
}

export interface ReopenIn {
  reason: string;
}

// ---------- pages / regions ----------
export interface PageInfo {
  page_no: number;
  width: number;
  height: number;
  flags: string[];
}

export interface PagesOut {
  pages: PageInfo[];
}

/** [x, y, w, h] in page-image pixels. */
export type BBox = [number, number, number, number];

export interface RegionOut {
  id: string;
  page_no: number;
  bbox: BBox;
  question_id: string | null;
  source: "auto" | "manual";
  is_blank: boolean;
  segmentation_confidence: number;
  flags: string[];
}

export type RegionAction = "crop" | "reassign" | "split" | "merge" | "blank" | "unblank";

export interface RegionPatch {
  action: RegionAction;
  bbox?: BBox | null;
  question_id?: string | null;
  split_y?: number | null;
  merge_with?: string | null;
  reason?: string | null;
}

export interface RegionCreate {
  page_no: number;
  bbox: BBox;
  question_id?: string | null;
}

export interface RegionEditOut {
  region: RegionOut | null;
  regions: RegionOut[];
  answers: AnswerOut[];
}

// ---------- audit ----------
export type AuditEntity = "answer" | "submission" | "exam" | "region";

export interface AuditOut {
  id: number;
  user_id: string | null;
  entity: string;
  entity_id: string;
  action: string;
  old_value: Record<string, unknown> | null;
  new_value: Record<string, unknown> | null;
  reason: string | null;
  created_at: string;
}

// ---------- system ----------
export interface CapabilitiesOut {
  ocr: { textlayer: boolean; trocr: boolean };
  graders: { lexical: boolean; sbert: boolean; nli: boolean };
  settings: {
    grader: string;
    ocr_engine: string;
    tau: number;
    tau_low: number;
    contradiction_threshold: number;
    auto_accept_threshold: number;
    max_upload_mb: number;
    max_pages: number;
  };
}

export interface HealthOut {
  status: string;
}

export interface ReadyOut {
  status: string;
  database: string;
  worker: "running" | "disabled" | string;
}
