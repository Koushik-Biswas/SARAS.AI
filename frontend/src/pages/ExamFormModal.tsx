import { useEffect, useState, type FormEvent } from "react";
import { ApiError, errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { ExamOut } from "../api/types";
import { Banner, Button, Field, Input, Modal, NumberInput, useToast } from "../components";

interface ExamFormModalProps {
  open: boolean;
  onClose: () => void;
  /** Present = edit mode. */
  exam?: ExamOut | null;
  onSaved: (exam: ExamOut) => void;
}

/** Create / edit exam metadata (title, subject, date, total marks). */
export function ExamFormModal({ open, onClose, exam, onSaved }: ExamFormModalProps) {
  const toast = useToast();
  const editing = !!exam;
  const [title, setTitle] = useState("");
  const [subject, setSubject] = useState("");
  const [date, setDate] = useState("");
  const [total, setTotal] = useState<number | null>(100);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setTitle(exam?.title ?? "");
    setSubject(exam?.subject ?? "");
    setDate(exam?.date ?? "");
    setTotal(exam?.total_marks ?? 100);
    setErrors({});
    setFormError(null);
  }, [open, exam]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const local: Record<string, string> = {};
    if (!title.trim()) local.title = "Give the exam a title.";
    const rubricMarks = exam?.rubric_marks ?? 0;
    if (total === null || total <= 0) local.total_marks = "Total marks must be greater than 0.";
    else if (total > 10000) local.total_marks = "Total marks must be 10,000 or less.";
    // The rubric could no longer be saved if its questions added up to more than the total.
    else if (editing && total < rubricMarks - 0.01) local.total_marks = `The rubric's questions already add up to ${rubricMarks}; the total can't be lower.`;
    setErrors(local);
    if (Object.keys(local).length) return;
    setBusy(true);
    setFormError(null);
    const body = { title: title.trim(), subject: subject.trim() || null, date: date || null, total_marks: total as number };
    try {
      const saved = editing ? await api.updateExam(exam!.id, body) : await api.createExam(body);
      toast.success(editing ? "Exam updated" : "Exam created", saved.title);
      onSaved(saved);
    } catch (err) {
      if (err instanceof ApiError && err.status === 400 && Object.keys(err.fieldErrors).length) {
        setErrors(err.fieldErrors);
      } else {
        setFormError(errorMessage(err));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      dismissible={!busy}
      title={editing ? "Edit exam" : "New exam"}
      description={editing ? "Update the exam details. The rubric is edited on the Rubric tab." : "Start with the basics. You'll add questions and the marking rubric next."}
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" type="submit" form="exam-form" loading={busy}>
            {editing ? "Save changes" : "Create exam"}
          </Button>
        </>
      }
    >
      <form id="exam-form" className="form-grid" onSubmit={submit} noValidate>
        {formError ? <Banner tone="danger" title="Could not save" className="form-grid__full">{formError}</Banner> : null}
        <Field label="Title" required error={errors.title} className="form-grid__full">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} placeholder="e.g. Biology mid-term, Class 10" />
        </Field>
        <Field label="Subject" error={errors.subject} hint="Optional">
          <Input value={subject} onChange={(e) => setSubject(e.target.value)} maxLength={120} placeholder="e.g. Biology" />
        </Field>
        <Field label="Exam date" error={errors.date} hint="Optional">
          <Input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </Field>
        <Field
          label="Total marks"
          required
          error={errors.total_marks}
          hint={editing && exam && exam.rubric_marks > 0 ? `At least ${exam.rubric_marks}, what the rubric's questions add up to.` : "The rubric's question marks should add up to this."}
        >
          <NumberInput value={total} onValueChange={setTotal} min={editing && exam ? exam.rubric_marks : 0} max={10000} />
        </Field>
      </form>
    </Modal>
  );
}
