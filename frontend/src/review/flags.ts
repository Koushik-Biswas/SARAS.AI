// Human-readable text for the machine flag codes the pipeline attaches to answers, regions,
// pages and sheets. Every flag renders as icon + text (never colour alone).
import {
  Ban,
  Calculator,
  CircleDashed,
  CircleMinus,
  Copy,
  Crop,
  FileWarning,
  Flag,
  Gauge,
  ImageOff,
  Info,
  Keyboard,
  Layers,
  ListChecks,
  ListOrdered,
  ScanText,
  SearchX,
  Sigma,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import type { QuestionType } from "../api/types";
import type { Tone } from "../components";

export interface FlagSpec {
  label: string;
  tone: Tone;
  icon: LucideIcon;
  /** Longer explanation (tooltip / help text). */
  hint?: string;
}

const FLAGS: Record<string, FlagSpec> = {
  // ---- answer-level
  low_ocr_confidence: {
    label: "Low OCR confidence",
    tone: "warning",
    icon: ScanText,
    hint: "The handwriting was hard to read. Compare the text with the scan and correct it.",
  },
  low_confidence: {
    label: "Low overall confidence",
    tone: "warning",
    icon: Gauge,
    hint: "The combined OCR, grading and segmentation confidence is below the auto-accept threshold.",
  },
  segmentation_warning: {
    label: "Answer boundaries uncertain",
    tone: "warning",
    icon: Crop,
    hint: "Check that the highlighted region covers the whole answer and nothing else.",
  },
  contradiction: {
    label: "Contains a contradicting statement",
    tone: "danger",
    icon: Ban,
    hint: "A sentence in the answer contradicts a key point of the model answer.",
  },
  needs_transcription: {
    label: "Handwriting not recognised — please type the answer",
    tone: "danger",
    icon: Keyboard,
    hint: "No OCR engine could read this region. Type what the student wrote, then save.",
  },
  copied_question: {
    label: "Answer repeats the question",
    tone: "warning",
    icon: Copy,
    hint: "Much of the answer is copied from the question prompt.",
  },
  keyword_list: {
    label: "Looks like a list of keywords",
    tone: "warning",
    icon: ListChecks,
    hint: "The answer lists key terms without explaining them.",
  },
  partial_credit: {
    label: "Partial credit awarded",
    tone: "info",
    icon: CircleMinus,
    hint: "At least one key point was only partly covered.",
  },
  graded_with_fallback: {
    label: "Graded with the fallback grader",
    tone: "info",
    icon: Info,
    hint: "The semantic grader was unavailable, so lexical matching was used.",
  },
  maths_manual_review: {
    label: "Maths answer — mark manually",
    tone: "warning",
    icon: Sigma,
    hint: "Worked maths is not auto-graded. Award marks by hand.",
  },
  missing_marker: {
    label: "Question number not found on the sheet",
    tone: "warning",
    icon: SearchX,
    hint: "No region is linked to this question. Draw one in the region editor if the answer exists.",
  },
  short_answer: {
    label: "Very short answer",
    tone: "warning",
    icon: ScanText,
    hint: "Only one or two characters were read. Check them against the scan before accepting the marks.",
  },
  marks_above_max: {
    label: "Marks above the question's new maximum",
    tone: "danger",
    icon: Sigma,
    hint: "The rubric's maximum was lowered after this answer was reviewed. Set the marks again.",
  },
  teacher_flag: { label: "Flagged by teacher", tone: "accent", icon: Flag },
  numeric_mismatch: {
    label: "Numbers differ from the model answer",
    tone: "warning",
    icon: Calculator,
  },
  no_rubric: { label: "No key points in the rubric", tone: "neutral", icon: ListChecks },
  ai_unavailable: {
    label: "AI could not mark this answer",
    tone: "danger",
    icon: TriangleAlert,
    hint: "No AI engine (Groq / Gemini) answered. Regrade later or enter the marks by hand.",
  },
  ai_off: {
    label: "AI marking is switched off",
    tone: "danger",
    icon: TriangleAlert,
    hint: "Turn on SARAS_AI_GRADING=ensemble and SARAS_AI_CONSENT=true in backend/.env, restart, then regrade.",
  },
  ai_uncertain: {
    label: "AI was unsure",
    tone: "warning",
    icon: Gauge,
    hint: "The AI marked this answer but was not confident. Check the marks.",
  },

  // ---- region / segmentation
  duplicate_marker: { label: "Question number appears twice", tone: "warning", icon: Copy },
  neighbour_missing: { label: "Neighbouring question not found", tone: "warning", icon: ListOrdered },
  unassigned: { label: "Text not linked to a question", tone: "warning", icon: CircleDashed },
  page_failed: { label: "A page could not be read", tone: "danger", icon: FileWarning },
  next_page_failed: {
    label: "Continues onto a page that failed",
    tone: "warning",
    icon: FileWarning,
    hint: "The next page could not be read, so this answer may be cut short.",
  },
  continues_across_pages: { label: "Continues across pages", tone: "neutral", icon: Layers },
  out_of_order: { label: "Answered out of order", tone: "neutral", icon: ListOrdered },
  partial_text: { label: "Only part of the text was read", tone: "neutral", icon: ScanText },

  // ---- page / sheet
  blank_page: { label: "Blank page", tone: "neutral", icon: CircleDashed },
  poor_quality: { label: "Poor scan quality", tone: "warning", icon: ImageOff },
  no_markers_found: { label: "No question numbers found", tone: "warning", icon: SearchX },
  unassigned_text: { label: "Text before the first question number", tone: "warning", icon: CircleDashed },
  gap_in_sequence: { label: "Question numbers skip", tone: "warning", icon: ListOrdered },
  has_failed_pages: { label: "Some pages failed", tone: "danger", icon: FileWarning },
};

export function flagSpec(code: string): FlagSpec {
  return (
    FLAGS[code] ?? {
      label: code.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()),
      tone: "neutral",
      icon: TriangleAlert,
    }
  );
}

/** Flags that only inform; they never put an answer into review on their own. */
export const INFO_FLAGS = new Set(["continues_across_pages", "out_of_order", "partial_text", "partial_credit", "graded_with_fallback"]);

export const QUESTION_TYPE_LABEL: Record<QuestionType, string> = {
  short: "Short answer",
  long: "Long answer",
  paragraph: "Paragraph",
  maths: "Maths",
};

export function questionTypeLabel(t: string): string {
  return (QUESTION_TYPE_LABEL as Record<string, string>)[t] ?? t;
}

/** Marks with at most 2 decimals and no trailing zeros ("4.5", "3", "2.25"). */
export function fmtMarks(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  return String(Math.round(n * 100) / 100);
}

export function pct(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  return `${Math.round(Math.max(0, Math.min(1, n)) * 100)}%`;
}
