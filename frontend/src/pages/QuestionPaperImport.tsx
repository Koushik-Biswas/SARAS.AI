// Step 1, second way in: upload the question paper and let the AI read the questions and
// marks into the rubric editor. The result is only a draft until the teacher saves it.
import { useRef, useState, type DragEvent } from "react";
import { FileText, Keyboard, ScanText, Sparkles, Upload } from "lucide-react";
import { errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { QuestionPaperOut } from "../api/types";
import { Banner, Button, ConfirmDialog, Spinner, cx } from "../components";

const ACCEPT = "application/pdf,.pdf,image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp";
const MAX_MB = 15;

export type RubricMode = "manual" | "paper";

function sourceLabel(src: string): string {
  if (src === "text") return "the PDF's text";
  const name = src.replace(/^ai:/, "");
  return `AI (${name.charAt(0).toUpperCase()}${name.slice(1)})`;
}

interface Props {
  examId: string;
  mode: RubricMode;
  onMode: (m: RubricMode) => void;
  disabled: boolean;
  /** Questions currently in the editor (replacing them needs a confirmation). */
  existing: number;
  onExtracted: (r: QuestionPaperOut) => void;
}

export function QuestionPaperImport({ examId, mode, onMode, disabled, existing, onExtracted }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [reading, setReading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [last, setLast] = useState<{ name: string; out: QuestionPaperOut } | null>(null);
  const [pending, setPending] = useState<File | null>(null);

  const read = async (file: File) => {
    setPending(null);
    setError(null);
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`${file.name} is larger than ${MAX_MB} MB.`);
      return;
    }
    setReading(file.name);
    try {
      const out = await api.extractQuestions(examId, file);
      setLast({ name: file.name, out });
      onExtracted(out);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setReading(null);
    }
  };
  const pick = (file: File | undefined) => {
    if (!file || disabled || reading) return;
    if (existing > 0) setPending(file);
    else void read(file);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  };

  return (
    <section className="rubric-mode" aria-label="How to set the questions">
      <div className="rubric-mode__options" role="radiogroup" aria-label="Question source">
        <button type="button" role="radio" aria-checked={mode === "manual"} className={cx("mode-tile", mode === "manual" && "is-active")} onClick={() => onMode("manual")}>
          <span className="mode-tile__icon" aria-hidden="true">
            <Keyboard size={20} />
          </span>
          <span className="mode-tile__text">
            <span className="mode-tile__title">Question settings</span>
            <span className="mode-tile__sub">Type each question and its marks</span>
          </span>
        </button>
        <button type="button" role="radio" aria-checked={mode === "paper"} className={cx("mode-tile", mode === "paper" && "is-active")} onClick={() => onMode("paper")}>
          <span className="mode-tile__icon mode-tile__icon--ai" aria-hidden="true">
            <ScanText size={20} />
          </span>
          <span className="mode-tile__text">
            <span className="mode-tile__title">
              Upload question sheet <span className="mode-tile__tag">AI</span>
            </span>
            <span className="mode-tile__sub">AI reads the questions and marks for you</span>
          </span>
        </button>
      </div>

      {mode === "paper" ? (
        <div
          className={cx("dropzone dropzone--compact", dragging && "is-dragging", (disabled || reading) && "is-disabled")}
          onDragOver={(e) => {
            e.preventDefault();
            if (!disabled && !reading) setDragging(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
          }}
          onDrop={onDrop}
        >
          {reading ? (
            <div className="paper-reading" role="status">
              <span className="paper-reading__scan" aria-hidden="true">
                <FileText size={30} />
                <span className="paper-reading__beam" />
              </span>
              <p className="dropzone__title">AI is reading the question paper</p>
              <p className="dropzone__text">
                <Spinner size={14} /> {reading}: finding each question, its type and marks...
              </p>
            </div>
          ) : (
            <>
              <div className="dropzone__icon" aria-hidden="true">
                <Sparkles size={26} />
              </div>
              <p className="dropzone__title">Drop the question paper here</p>
              <p className="dropzone__text">PDF, or a photo / scan (PNG, JPG, WEBP) · up to {MAX_MB} MB and 10 pages</p>
              <Button variant="primary" iconLeft={<Upload size={16} aria-hidden="true" />} disabled={disabled} onClick={() => inputRef.current?.click()}>
                Choose question paper
              </Button>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                className="sr-only"
                tabIndex={-1}
                aria-hidden="true"
                onChange={(e) => {
                  pick(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </>
          )}
        </div>
      ) : null}

      {mode === "paper" && disabled ? (
        <p className="subtle">Answer sheets are already uploaded, so the question list is fixed. Edit the questions below instead.</p>
      ) : null}
      {error ? (
        <Banner tone="danger" title="The question paper could not be read">
          {error}
        </Banner>
      ) : null}
      {last && mode === "paper" ? (
        <Banner tone="success" title={`${last.out.questions.length} question${last.out.questions.length === 1 ? "" : "s"} read from ${last.name}`}>
          Read with {sourceLabel(last.out.source)}. Check every question below, fix anything the AI got wrong, then <strong>Save rubric</strong>.
          {last.out.warnings.length ? (
            <ul className="plain-list paper-warnings">
              {last.out.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          ) : null}
        </Banner>
      ) : null}

      <ConfirmDialog
        open={!!pending}
        title="Replace the questions in the editor?"
        message={
          <>
            The {existing} question{existing === 1 ? "" : "s"} now in the editor will be replaced by the questions read from <strong>{pending?.name}</strong>. Nothing is saved until you click Save rubric.
          </>
        }
        confirmLabel="Read and replace"
        onConfirm={() => pending && void read(pending)}
        onCancel={() => setPending(null)}
      />
    </section>
  );
}
