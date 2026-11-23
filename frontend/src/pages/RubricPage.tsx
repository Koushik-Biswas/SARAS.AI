import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, CircleExclamationPoint, Lock, Plus, RefreshCw, Save, Sparkles, Trash, TrendingUp, Undo2 } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { QuestionIn, QuestionOut, QuestionPaperOut, QuestionType } from "../api/types";
import {
  Badge,
  Banner,
  Button,
  Card,
  EmptyState,
  ErrorBanner,
  Field,
  IconButton,
  NumberInput,
  Select,
  Skeleton,
  Textarea,
  cx,
  useToast,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { canEditExam, useExam } from "./ExamLayout";
import { QuestionPaperImport, type RubricMode } from "./QuestionPaperImport";
import { StepFooter } from "./workflow";

// ---------------------------------------------------------------- draft model
// A question is its number, type, max marks and text. There is no model answer and no
// key points: the AI engines (Groq / Gemini) read each answer and give the marks.
interface QuestionDraft {
  uid: string;
  question_no: number | null;
  type: QuestionType;
  max_marks: number | null;
  prompt: string;
  open: boolean;
}

const TYPES: { value: QuestionType; label: string }[] = [
  { value: "short", label: "Short answer" },
  { value: "long", label: "Long answer" },
  { value: "paragraph", label: "Paragraph / essay" },
  { value: "maths", label: "Maths / numeric" },
];

let uidSeq = 0;
const uid = () => `d${Date.now().toString(36)}${(uidSeq++).toString(36)}`;
const round2 = (n: number) => Math.round(n * 100) / 100;

function fromServer(qs: QuestionOut[]): QuestionDraft[] {
  return [...qs]
    .sort((a, b) => a.question_no - b.question_no)
    .map((q) => ({
      uid: uid(),
      question_no: q.question_no,
      type: q.type,
      max_marks: q.max_marks,
      prompt: q.prompt ?? "",
      open: false,
    }));
}

function newQuestion(no: number): QuestionDraft {
  return {
    uid: uid(),
    question_no: no,
    type: "short",
    max_marks: 5,
    prompt: "",
    open: true,
  };
}

/** Drafts from a question paper read by the AI (or the PDF text reader). */
function fromPaper(out: QuestionPaperOut): QuestionDraft[] {
  return out.questions.map((q, i) => ({
    uid: uid(),
    question_no: q.question_no,
    type: q.type,
    max_marks: q.max_marks,
    prompt: q.prompt,
    open: i === 0 || q.max_marks === null,
  }));
}

function toPayload(drafts: QuestionDraft[]): QuestionIn[] {
  return drafts.map((d) => ({
    question_no: d.question_no ?? 0,
    type: d.type,
    max_marks: d.max_marks ?? 0,
    prompt: d.prompt.trim() || null,
    model_answer: null,
    key_points: [],
  }));
}

interface QuestionIssues {
  question?: string[];
  question_no?: string;
  max_marks?: string;
  prompt?: string;
}

function validate(drafts: QuestionDraft[]): QuestionIssues[] {
  const nums = drafts.map((d) => d.question_no);
  return drafts.map((d) => {
    const iss: QuestionIssues = {};
    if (d.question_no === null || d.question_no < 1 || d.question_no > 99 || !Number.isInteger(d.question_no)) {
      iss.question_no = "Use a whole number from 1 to 99.";
    } else if (nums.filter((n) => n === d.question_no).length > 1) {
      iss.question_no = `Q${d.question_no} is used more than once.`;
    }
    if (d.max_marks === null || d.max_marks <= 0) iss.max_marks = "Must be greater than 0.";
    else if (d.max_marks > 1000) iss.max_marks = "Must be 1000 or less.";
    if (!d.prompt.trim()) iss.prompt = "Type the question: the AI marks the answer against it.";
    return iss;
  });
}

const hasIssues = (i: QuestionIssues) => !!(i.question?.length || i.question_no || i.max_marks || i.prompt);

/** Map server 400 details ("questions.2.key_points.1.weight") onto question indices. */
function mapServerErrors(err: ApiError): { byIndex: Record<number, string[]>; general: string[] } {
  const byIndex: Record<number, string[]> = {};
  const general: string[] = [];
  const details = Array.isArray(err.details) ? err.details : [];
  for (const d of details as { field?: string; message?: string }[]) {
    const m = /^questions\.(\d+)(?:\.(.+))?$/.exec(d.field ?? "");
    const msg = (d.message ?? "Invalid value").replace(/^Value error, /, "");
    if (m) {
      const i = Number(m[1]);
      (byIndex[i] ??= []).push(m[2] ? `${m[2].replace(/_/g, " ")}: ${msg}` : msg);
    } else general.push(d.field && d.field !== "questions" ? `${d.field}: ${msg}` : msg);
  }
  if (!details.length) general.push(err.message);
  return { byIndex, general };
}

// ---------------------------------------------------------------- question editor
interface EditorProps {
  draft: QuestionDraft;
  index: number;
  count: number;
  issues: QuestionIssues;
  serverErrors?: string[];
  showErrors: boolean;
  structureLocked: boolean;
  onChange: (d: QuestionDraft) => void;
  onMove: (dir: -1 | 1) => void;
  onRemove: () => void;
}

function QuestionEditor({ draft, index, count, issues, serverErrors, showErrors, structureLocked, onChange, onMove, onRemove }: EditorProps) {
  const set = <K extends keyof QuestionDraft>(k: K, v: QuestionDraft[K]) => onChange({ ...draft, [k]: v });
  const invalid = hasIssues(issues) || !!serverErrors?.length;
  const headId = `q-${draft.uid}-head`;
  const panelId = `q-${draft.uid}-panel`;
  const err = (v?: string) => (showErrors ? v : undefined);

  return (
    <div className={cx("q-editor", draft.open && "is-open", invalid && showErrors && "has-error")}>
      <div className="q-editor__head">
        <button
          type="button"
          id={headId}
          className="q-editor__toggle"
          aria-expanded={draft.open}
          aria-controls={panelId}
          onClick={() => set("open", !draft.open)}
        >
          {draft.open ? <ChevronDown size={18} aria-hidden="true" /> : <ChevronRight size={18} aria-hidden="true" />}
          <span className="q-editor__no num">Q{draft.question_no ?? "?"}</span>
          <span className="q-editor__summary truncate">{draft.prompt.trim() || <span className="subtle">No question text</span>}</span>
        </button>
        <div className="q-editor__chips">
          <Badge size="sm" tone="neutral">
            {TYPES.find((t) => t.value === draft.type)?.label ?? draft.type}
          </Badge>
          <span className="q-editor__marks num">
            {draft.max_marks ?? "-"} <span className="subtle">marks</span>
          </span>
          <Badge size="sm" tone="info" icon={<Sparkles size={12} aria-hidden="true" />}>
            AI marked
          </Badge>
        </div>
        <div className="q-editor__tools">
          <IconButton size="sm" label={`Move Q${draft.question_no ?? ""} up`} icon={<ArrowUp size={16} aria-hidden="true" />} disabled={structureLocked || index === 0} onClick={() => onMove(-1)} />
          <IconButton size="sm" label={`Move Q${draft.question_no ?? ""} down`} icon={<ArrowDown size={16} aria-hidden="true" />} disabled={structureLocked || index === count - 1} onClick={() => onMove(1)} />
          <IconButton size="sm" label={`Remove Q${draft.question_no ?? ""}`} icon={<Trash size={16} aria-hidden="true" />} disabled={structureLocked} onClick={onRemove} />
        </div>
      </div>

      {showErrors && (issues.question?.length || serverErrors?.length) ? (
        <ul className="q-editor__errors" role="alert">
          {[...(issues.question ?? []), ...(serverErrors ?? [])].map((m, i) => (
            <li key={i}>
              <CircleExclamationPoint size={14} aria-hidden="true" /> {m}
            </li>
          ))}
        </ul>
      ) : null}

      {draft.open ? (
        <div className="q-editor__body" id={panelId} role="region" aria-labelledby={headId}>
          <div className="q-grid">
            <Field label="No." error={err(issues.question_no)} hint={structureLocked ? "Fixed after upload" : undefined}>
              <NumberInput value={draft.question_no} decimals={0} min={1} max={99} disabled={structureLocked} onValueChange={(v) => set("question_no", v)} />
            </Field>
            <Field label="Type">
              <Select value={draft.type} onChange={(e) => set("type", e.target.value as QuestionType)}>
                {TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Max marks" error={err(issues.max_marks)}>
              <NumberInput value={draft.max_marks} min={0} max={1000} onValueChange={(v) => set("max_marks", v)} />
            </Field>
          </div>
          <Field label="Question" error={err(issues.prompt)} hint="The AI engines read each student's answer to this question and give marks out of the max marks.">
            <Textarea rows={3} value={draft.prompt} maxLength={5000} onChange={(e) => set("prompt", e.target.value)} placeholder="e.g. What is the national animal of India?" />
          </Field>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- page
export default function RubricPage() {
  const { exam, reload: reloadExam, setExam } = useExam();
  const canEdit = canEditExam(exam); // only the owner saves the rubric (admins read others' exams)
  const [mode, setMode] = useState<RubricMode>(exam.stats.questions > 0 ? "manual" : "paper");
  const [raising, setRaising] = useState(false);
  const toast = useToast();
  const q = useAsync(() => api.getQuestions(exam.id), [exam.id]);
  const [drafts, setDrafts] = useState<QuestionDraft[]>([]);
  const [baseline, setBaseline] = useState("");
  const [showErrors, setShowErrors] = useState(false);
  const [serverErrors, setServerErrors] = useState<{ byIndex: Record<number, string[]>; general: string[] } | null>(null);
  const [saving, setSaving] = useState(false);
  const [offerRegrade, setOfferRegrade] = useState(false);
  const [regrading, setRegrading] = useState(false);
  const loadedFor = useRef<QuestionOut[] | undefined>(undefined);

  useEffect(() => {
    if (q.data && loadedFor.current !== q.data) {
      loadedFor.current = q.data;
      const d = fromServer(q.data);
      if (d.length === 1) d[0].open = true;
      setDrafts(d);
      setBaseline(JSON.stringify(toPayload(d)));
    }
  }, [q.data]);

  const structureLocked = exam.stats.sheets > 0 && (q.data?.length ?? 0) > 0;
  const issues = useMemo(() => validate(drafts), [drafts]);
  const anyIssues = issues.some(hasIssues);
  const rubricTotal = round2(drafts.reduce((a, d) => a + (d.max_marks ?? 0), 0));
  const overTotal = rubricTotal > exam.total_marks + 0.01;
  const dirty = JSON.stringify(toPayload(drafts)) !== baseline;
  const canSave = canEdit && drafts.length > 0 && !anyIssues && !overTotal && dirty;

  const fromQuestionPaper = (out: QuestionPaperOut) => {
    setDrafts(fromPaper(out));
    setServerErrors(null);
    setShowErrors(true);
  };

  const raiseTotal = async () => {
    setRaising(true);
    try {
      const updated = await api.updateExam(exam.id, { total_marks: rubricTotal });
      setExam(updated);
      toast.success("Exam total updated", `The exam is now out of ${rubricTotal} marks.`);
    } catch (err) {
      toast.error("Could not change the exam total", errorMessage(err));
    } finally {
      setRaising(false);
    }
  };

  const update = (i: number, d: QuestionDraft) => {
    setDrafts((xs) => xs.map((x, j) => (j === i ? d : x)));
    setServerErrors(null);
  };
  const move = (i: number, dir: -1 | 1) =>
    setDrafts((xs) => {
      const j = i + dir;
      if (j < 0 || j >= xs.length) return xs;
      const out = [...xs];
      const a = { ...out[i], question_no: out[j].question_no };
      const b = { ...out[j], question_no: out[i].question_no };
      out[i] = b;
      out[j] = a;
      return out;
    });
  const add = () =>
    setDrafts((xs) => [...xs.map((x) => ({ ...x, open: false })), newQuestion(Math.min(99, Math.max(0, ...xs.map((x) => x.question_no ?? 0)) + 1))]);

  const save = async () => {
    setShowErrors(true);
    if (!canSave) return;
    setSaving(true);
    setServerErrors(null);
    try {
      const saved = await api.saveQuestions(exam.id, { questions: toPayload(drafts) });
      loadedFor.current = saved;
      const openByNo = new Set(drafts.filter((d) => d.open).map((d) => d.question_no));
      const next = fromServer(saved).map((d) => ({ ...d, open: openByNo.has(d.question_no) }));
      setDrafts(next);
      setBaseline(JSON.stringify(toPayload(next)));
      q.setData(saved);
      setShowErrors(false);
      toast.success("Rubric saved", `${saved.length} question${saved.length === 1 ? "" : "s"}, ${round2(saved.reduce((a, x) => a + x.max_marks, 0))} marks`);
      if (exam.stats.processed - exam.stats.approved > 0) setOfferRegrade(true);
      void reloadExam(true);
    } catch (err) {
      if (err instanceof ApiError && (err.status === 400 || err.status === 409)) {
        const mapped = err.status === 400 ? mapServerErrors(err) : { byIndex: {}, general: [err.message] };
        setServerErrors(mapped);
        setDrafts((xs) => xs.map((x, i) => (mapped.byIndex[i] ? { ...x, open: true } : x)));
        toast.error("Rubric not saved", "Fix the highlighted problems and try again.");
      } else {
        toast.error("Rubric not saved", errorMessage(err));
      }
    } finally {
      setSaving(false);
    }
  };

  const regrade = async () => {
    setRegrading(true);
    try {
      const r = await api.regradeExam(exam.id);
      toast.success("Sheets regraded", `${r.regraded} sheet${r.regraded === 1 ? "" : "s"} re-marked with the new rubric.`);
      setOfferRegrade(false);
      void reloadExam(true);
    } catch (err) {
      toast.error("Regrade failed", errorMessage(err));
    } finally {
      setRegrading(false);
    }
  };

  if (q.loading && !q.data) {
    return (
      <Card>
        <div className="stack">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} height={52} radius={12} />
          ))}
        </div>
      </Card>
    );
  }
  if (q.error && !q.data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load the rubric" />;

  return (
    <div className="stack">
      {!canEdit ? (
        <Banner tone="info" title="View only">
          Only the teacher who owns this exam can change its rubric or regrade its sheets.
        </Banner>
      ) : null}
      {structureLocked ? (
        <Banner tone="info" title="Sheets are already uploaded">
          You can still edit question text and marks, then regrade. Questions can't be added, removed or renumbered.
        </Banner>
      ) : null}
      {offerRegrade ? (
        <Banner
          tone="success"
          title="Rubric saved. Regrade existing sheets?"
          action={
            <div className="row-sm">
              <Button size="sm" variant="ghost" onClick={() => setOfferRegrade(false)}>
                Later
              </Button>
              <Button size="sm" variant="primary" loading={regrading} iconLeft={<RefreshCw size={14} aria-hidden="true" />} onClick={regrade}>
                Regrade sheets
              </Button>
            </div>
          }
        >
          Unapproved sheets keep their old marks until you regrade. Approved sheets are never changed.
        </Banner>
      ) : null}
      {serverErrors?.general.length ? (
        <Banner tone="danger" title="The server rejected the rubric">
          <ul className="plain-list">
            {serverErrors.general.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </Banner>
      ) : null}

      {canEdit ? (
        <QuestionPaperImport
          examId={exam.id}
          mode={mode}
          onMode={setMode}
          disabled={structureLocked}
          existing={drafts.length}
          onExtracted={fromQuestionPaper}
        />
      ) : null}

      <Card
        title={mode === "paper" ? "Questions read from the paper" : "Question settings"}
        subtitle={
          mode === "paper"
            ? "Check what the AI read: question text, type and marks. Edit anything, then save."
            : "Add each question and its marks. The AI engines read the answer sheets and give the marks."
        }
        actions={
          <Button variant="secondary" iconLeft={<Plus size={16} aria-hidden="true" />} onClick={add} disabled={!canEdit || structureLocked || drafts.length >= 99}>
            Add question
          </Button>
        }
      >
        {drafts.length === 0 ? (
          <EmptyState
            compact
            illustration="review"
            title="No questions yet"
            description={
              mode === "paper"
                ? "Upload the question paper above and the AI fills in every question and its marks. Or add them yourself."
                : "Add each question with its marks. The AI marks every answer against it."
            }
            action={
              canEdit ? (
                <Button variant={mode === "paper" ? "secondary" : "primary"} iconLeft={<Plus size={16} aria-hidden="true" />} onClick={add}>
                  Add first question
                </Button>
              ) : undefined
            }
          />
        ) : (
          <div className="q-list">
            {drafts.map((d, i) => (
              <QuestionEditor
                key={d.uid}
                draft={d}
                index={i}
                count={drafts.length}
                issues={issues[i]}
                serverErrors={serverErrors?.byIndex[i]}
                showErrors={showErrors || dirty || !!serverErrors}
                structureLocked={structureLocked}
                onChange={(nd) => update(i, nd)}
                onMove={(dir) => move(i, dir)}
                onRemove={() => setDrafts((xs) => xs.filter((_, j) => j !== i))}
              />
            ))}
          </div>
        )}
      </Card>

      <div className="savebar" role="region" aria-label="Rubric summary">
        <div className="savebar__info">
          <span className={cx("savebar__total num", overTotal && "is-off")}>
            {overTotal ? <CircleExclamationPoint size={16} aria-hidden="true" /> : <Lock size={16} aria-hidden="true" className="subtle" />}
            Rubric total <strong>{rubricTotal}</strong> of {exam.total_marks} exam marks
          </span>
          {overTotal ? <span className="savebar__msg is-off">Questions add up to more than the exam's total marks.</span> : null}
          {overTotal && canEdit ? (
            <Button size="sm" variant="secondary" loading={raising} iconLeft={<TrendingUp size={14} aria-hidden="true" />} onClick={raiseTotal}>
              Make the exam out of {rubricTotal}
            </Button>
          ) : null}
          {!overTotal && anyIssues ? (
            <span className="savebar__msg is-warn">
              {issues.filter(hasIssues).length} question{issues.filter(hasIssues).length === 1 ? " needs" : "s need"} attention before saving.
            </span>
          ) : null}
          {!overTotal && !anyIssues && rubricTotal < exam.total_marks && drafts.length > 0 ? (
            <span className="savebar__msg">{round2(exam.total_marks - rubricTotal)} marks not yet assigned.</span>
          ) : null}
        </div>
        <div className="row-sm">
          {dirty ? (
            <Button
              variant="ghost"
              iconLeft={<Undo2 size={16} aria-hidden="true" />}
              onClick={() => {
                if (q.data) {
                  const d = fromServer(q.data);
                  setDrafts(d);
                  setServerErrors(null);
                  setShowErrors(false);
                }
              }}
            >
              Discard changes
            </Button>
          ) : null}
          <Button variant="primary" iconLeft={<Save size={16} aria-hidden="true" />} loading={saving} disabled={!canSave} onClick={save}>
            {dirty ? "Save rubric" : "Saved"}
          </Button>
        </div>
      </div>
      <StepFooter exam={exam} slug="rubric" ready={exam.stats.questions > 0 && !dirty} hint={dirty ? "Save the rubric before moving on" : undefined} />
    </div>
  );
}
