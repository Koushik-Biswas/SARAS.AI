// Right-hand side of the review workspace: question, flags, AI assessment, editable OCR
// text, key points and the marking form (marks / reason / comment).
import { useId, type RefObject } from "react";
import { CircleDot, Crop, Lock, PencilLine, RotateCcw } from "lucide-react";
import type { AnswerOut } from "../api/types";
import { Badge, Banner, Button, ButtonLink, ConfidenceMeter, Field, NumberInput, StatusPill, Textarea, cx } from "../components";
import { KeyPointList } from "./KeyPointList";
import { INFO_FLAGS, flagSpec, fmtMarks, pct, questionTypeLabel } from "./flags";

// ---------------------------------------------------------------- draft model
export interface Draft {
  text: string;
  marks: number | null;
  comment: string;
  reason: string;
}

/** Marks the form starts from: an earlier teacher decision, else the AI's marks. */
export function initialMarks(a: AnswerOut): number {
  return a.final_marks ?? a.ai_marks;
}

export function draftFrom(a: AnswerOut): Draft {
  return { text: a.ocr_text ?? "", marks: initialMarks(a), comment: a.comment ?? "", reason: "" };
}

const eq = (x: number | null, y: number | null) => x !== null && y !== null && Math.abs(x - y) < 1e-9;

export interface DraftStatus {
  textChanged: boolean;
  marksChanged: boolean;
  commentChanged: boolean;
  dirty: boolean;
  /** Mirrors the API rule: a reason is needed when the marks differ from both the AI and the current final marks. */
  reasonRequired: boolean;
}

export function draftStatus(a: AnswerOut, d: Draft): DraftStatus {
  const textChanged = d.text.trim() !== (a.ocr_text ?? "").trim();
  const marksChanged = !eq(d.marks, initialMarks(a));
  const commentChanged = d.comment.trim() !== (a.comment ?? "").trim();
  const reasonRequired = marksChanged && d.marks !== null && !eq(d.marks, a.ai_marks) && !eq(d.marks, a.final_marks);
  return {
    textChanged,
    marksChanged,
    commentChanged,
    dirty: textChanged || marksChanged || commentChanged || d.reason.trim() !== "",
    reasonRequired,
  };
}

/** Client-side checks before approving; returns field errors keyed like the API's. */
export function validateDraft(a: AnswerOut, d: Draft, approving: boolean): Record<string, string> {
  const errs: Record<string, string> = {};
  const st = draftStatus(a, d);
  if (approving) {
    if (d.marks === null) errs.final_marks = "Enter the marks to award.";
    else if (d.marks < 0 || d.marks > a.max_marks + 1e-9) errs.final_marks = `Marks must be between 0 and ${fmtMarks(a.max_marks)}.`;
    else if (Math.abs(d.marks * 100 - Math.round(d.marks * 100)) > 1e-6) errs.final_marks = "Use at most 2 decimal places.";
    if (st.reasonRequired && !d.reason.trim()) errs.reason = "Give a reason for changing the AI's marks.";
  }
  if (d.text.length > 20000) errs.corrected_text = "The text is too long (20,000 characters max).";
  return errs;
}

// ---------------------------------------------------------------- panel
interface AnswerPanelProps {
  answer: AnswerOut;
  draft: Draft;
  onDraft: (patch: Partial<Draft>) => void;
  errors: Record<string, string>;
  readOnly: boolean;
  autoAccept: number;
  contradictionThreshold: number;
  regionsHref: string;
  textRef: RefObject<HTMLTextAreaElement>;
  marksRef: RefObject<HTMLInputElement>;
  reasonRef: RefObject<HTMLTextAreaElement>;
}

export function AnswerPanel({
  answer: a,
  draft,
  onDraft,
  errors,
  readOnly,
  autoAccept,
  contradictionThreshold,
  regionsHref,
  textRef,
  marksRef,
  reasonRef,
}: AnswerPanelProps) {
  const headingId = useId();
  const st = draftStatus(a, draft);
  const flags = a.flags ?? [];
  const reviewFlags = flags.filter((f) => !INFO_FLAGS.has(f));
  const infoFlags = flags.filter((f) => INFO_FLAGS.has(f));
  const needsTranscription = flags.includes("needs_transcription");
  const missingRegion = a.region_ids.length === 0;
  const matched = a.key_points.filter((k) => k.status === "matched").length;
  const differsFromAi = draft.marks !== null && Math.abs(draft.marks - a.ai_marks) > 1e-9;

  return (
    <article className="ap" aria-labelledby={headingId}>
      <header className="ap__head">
        <div className="ap__title-row">
          <h2 id={headingId} className="ap__qno">
            Question <span className="num">{a.question_no}</span>
          </h2>
          <Badge size="sm" tone="neutral">
            {questionTypeLabel(a.question_type)}
          </Badge>
          <StatusPill kind="answer" status={a.status} size="sm" />
          {a.locked ? (
            <Badge size="sm" tone="success" icon={<Lock size={12} aria-hidden="true" />}>
              Locked
            </Badge>
          ) : null}
        </div>
        {a.question_prompt ? <p className="ap__prompt">{a.question_prompt}</p> : null}
      </header>

      {flags.length ? (
        <section className="ap__flags" aria-label="Review flags">
          <ul className="chip-list">
            {[...reviewFlags, ...infoFlags].map((f) => {
              const spec = flagSpec(f);
              const Icon = spec.icon;
              return (
                <li key={f}>
                  <Badge tone={spec.tone} title={spec.hint} icon={<Icon size={13} aria-hidden="true" />}>
                    {spec.label}
                    {spec.hint ? <span className="sr-only">. {spec.hint}</span> : null}
                  </Badge>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      <section className="score" aria-label="AI assessment">
        <div className="score__block">
          <span className="score__label">AI marks</span>
          <span className="score__value num">
            {fmtMarks(a.ai_marks)}
            <small> / {fmtMarks(a.max_marks)}</small>
          </span>
          {a.final_marks !== null ? (
            <span className="score__final">
              Teacher: <strong className="num">{fmtMarks(a.final_marks)}</strong>
            </span>
          ) : (
            <span className="score__final subtle">{a.grader ? `Graded by ${a.grader}` : "Not graded yet"}</span>
          )}
        </div>
        <div className="score__block score__block--grow">
          <span className="score__label">Confidence</span>
          <ConfidenceMeter value={a.confidence} tau={autoAccept} tauLow={Math.min(0.5, autoAccept)} label="Overall confidence" />
          <details className="conf-details">
            <summary>Breakdown</summary>
            <dl className="conf-details__list">
              <div>
                <dt>Handwriting (OCR)</dt>
                <dd>
                  <ConfidenceMeter value={a.ocr_confidence} tau={autoAccept} tauLow={Math.min(0.5, autoAccept)} label="OCR confidence" compact />
                </dd>
              </div>
              <div>
                <dt>Grading</dt>
                <dd>
                  <ConfidenceMeter value={a.grading_confidence} tau={autoAccept} tauLow={Math.min(0.5, autoAccept)} label="Grading confidence" compact />
                </dd>
              </div>
              <div>
                <dt>Answer boundaries</dt>
                <dd>
                  <ConfidenceMeter
                    value={a.segmentation_confidence}
                    tau={autoAccept}
                    tauLow={Math.min(0.5, autoAccept)}
                    label="Segmentation confidence"
                    compact
                  />
                </dd>
              </div>
            </dl>
            <p className="conf-details__note">
              Calibrated from all three. Answers below {pct(autoAccept)} or with a review flag are sent to you.
            </p>
          </details>
        </div>
      </section>

      <section className="ap__section" aria-label="Student answer">
        <div className="ap__section-head">
          <h3 className="ap__h">Student answer</h3>
          <div className="row-sm">
            {a.text_corrected ? (
              <Badge size="sm" tone="info" icon={<PencilLine size={12} aria-hidden="true" />}>
                Text corrected
              </Badge>
            ) : null}
            {st.textChanged ? (
              <Badge size="sm" tone="warning" icon={<CircleDot size={12} aria-hidden="true" />}>
                Edited · not saved
              </Badge>
            ) : null}
            <span className="ap__ocr subtle num" title="OCR confidence for this answer">
              OCR {pct(a.ocr_confidence)}
            </span>
          </div>
        </div>
        {needsTranscription && !readOnly ? (
          <Banner tone="warning" title="Handwriting not recognised">
            Type what the student wrote, using the scan on the left, then save or approve. The answer is re-graded from your text.
          </Banner>
        ) : null}
        {missingRegion ? (
          <Banner
            tone="info"
            title="No region is linked to this question"
            action={
              !readOnly ? (
                <ButtonLink to={regionsHref} size="sm" variant="secondary" iconLeft={<Crop size={14} aria-hidden="true" />}>
                  Edit regions
                </ButtonLink>
              ) : undefined
            }
          >
            If the student answered it, draw the region in the region editor so it can be read and graded.
          </Banner>
        ) : null}
        <Field
          label="Answer text"
          hideLabel
          hint={readOnly ? undefined : "Fix OCR mistakes here. Saving re-grades the answer against the key points."}
          error={errors.corrected_text}
        >
          <Textarea
            ref={textRef}
            className="ap__text"
            rows={7}
            value={draft.text}
            readOnly={readOnly}
            placeholder={a.status === "blank" ? "The student left this answer blank." : "No text was read for this answer."}
            onChange={(e) => onDraft({ text: e.target.value })}
          />
        </Field>
      </section>

      {a.key_points.length ? (
        <section className="ap__section" aria-label="Key points">
          <div className="ap__section-head">
            <h3 className="ap__h">Key points</h3>
            <span className="subtle num ap__kp-count">
              {matched} of {a.key_points.length} matched
            </span>
          </div>
          <KeyPointList points={a.key_points} contradictionThreshold={contradictionThreshold} />
        </section>
      ) : (
        <section className="ap__section" aria-label="AI assessment">
          <div className="ap__section-head">
            <h3 className="ap__h">AI assessment</h3>
            <span className="subtle num ap__kp-count">
              {a.grader ? `${fmtMarks(a.ai_marks)} / ${fmtMarks(a.max_marks)}` : "Not marked by AI"}
              {a.verdict ? ` · ${a.verdict.replace(/_/g, " ")}` : ""}
            </span>
          </div>
          <p className={a.feedback ? undefined : "subtle"}>
            {a.feedback ?? "The AI has not marked this answer. Check the flags above, then enter the marks."}
          </p>
        </section>
      )}

      <section className="ap__section ap__marking" aria-label="Marking">
        <h3 className="ap__h">Marking</h3>
        <div className="marks-row">
          <Field label="Final marks" error={errors.final_marks} hint={readOnly ? undefined : `0 to ${fmtMarks(a.max_marks)}`}>
            <NumberInput
              ref={marksRef}
              inputSize="lg"
              value={draft.marks}
              onValueChange={(v) => onDraft({ marks: v })}
              min={0}
              max={a.max_marks}
              step={0.5}
              decimals={2}
              disabled={readOnly}
              endAdornment={<span className="marks-suffix num">/ {fmtMarks(a.max_marks)}</span>}
            />
          </Field>
          <div className="marks-row__ai">
            <span className="muted">
              AI suggested <strong className="num">{fmtMarks(a.ai_marks)}</strong>
            </span>
            {differsFromAi && !readOnly ? (
              <Button size="sm" variant="ghost" iconLeft={<RotateCcw size={14} aria-hidden="true" />} onClick={() => onDraft({ marks: a.ai_marks, reason: "" })}>
                Use AI marks
              </Button>
            ) : null}
          </div>
        </div>
        <Field
          className={cx(st.reasonRequired && "field--attention")}
          label={st.reasonRequired ? "Reason for changing the marks" : "Reason (optional)"}
          required={st.reasonRequired}
          error={errors.reason}
          hint={
            readOnly
              ? undefined
              : st.reasonRequired
                ? `Required: your marks differ from the AI's ${fmtMarks(a.ai_marks)}. Saved in the audit log.`
                : "Saved in the audit log with your decision."
          }
        >
          <Textarea ref={reasonRef} rows={2} maxLength={1000} value={draft.reason} readOnly={readOnly} onChange={(e) => onDraft({ reason: e.target.value })} />
        </Field>
        <Field label="Comment (optional)" error={errors.comment}>
          <Textarea rows={2} maxLength={2000} value={draft.comment} readOnly={readOnly} onChange={(e) => onDraft({ comment: e.target.value })} />
        </Field>
      </section>
    </article>
  );
}
