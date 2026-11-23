// The core review screen for one student sheet (PRD 17.1 screen 6): answer queue, scan with
// region overlay, answer details + marking form, sticky action bar and keyboard shortcuts.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Check, ChevronDown, ChevronUp, Flag, Save } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { AnswerOut, ReviewIn, SubmissionOut } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import {
  Banner,
  Button,
  ButtonLink,
  Card,
  ConfirmDialog,
  EmptyState,
  ErrorBanner,
  IconButton,
  Kbd,
  ProgressBar,
  Skeleton,
  cx,
  useToast,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { usePolling } from "../hooks/usePolling";
import { useBreadcrumbs } from "../layout/ShellContext";
import { canEditExam, formatDate, useExam } from "../pages/ExamLayout";
import { stageLabel } from "../pages/stages";
import { AnswerPanel, draftFrom, draftStatus, validateDraft, type Draft } from "./AnswerPanel";
import { AnswerQueue, type QueueFilter } from "./AnswerQueue";
import { ShortcutHelp, SplitPane, zoomStep, type ShortcutDef } from "./controls";
import { fmtMarks } from "./flags";
import { inArrowWidget, isTypingTarget, modalOpen, useCapabilities, useHotkeys, useUnsavedGuard } from "./hooks";
import { questionLookup } from "./regions";
import { ScanViewer } from "./ScanViewer";
import { SheetHeader } from "./SheetHeader";

type Busy = null | "approve" | "save" | "flag";
type Nav =
  | { kind: "answer"; id: string }
  | { kind: "route"; to: string }
  // Held by the unsaved-changes guard: browser Back/Forward, a link, the sidebar, sign-out...
  | { kind: "resume"; resume: () => void; cancel: () => void };

const isPending = (a: AnswerOut) => a.status === "needs_review" || a.status === "failed";
const byRef = (a: SubmissionOut, b: SubmissionOut) => a.student_ref.localeCompare(b.student_ref, undefined, { numeric: true });

const SHORTCUTS: ShortcutDef[] = [
  { label: "Next answer", keys: [["J"], ["↓"]] },
  { label: "Previous answer", keys: [["K"], ["↑"]] },
  { label: "Approve answer", keys: [["A"], ["Ctrl", "Enter"]] },
  { label: "Save text without approving", keys: [["Ctrl", "S"]] },
  { label: "Flag for review", keys: [["F"]] },
  { label: "Edit the answer text", keys: [["E"]] },
  { label: "Zoom in / out / fit", keys: [["+"], ["−"], ["0"]] },
  { label: "Show this help", keys: [["?"]] },
];

/** Normalise API field paths ("body.final_marks" -> "final_marks"). */
function fieldErrorsOf(err: ApiError): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(err.fieldErrors)) out[k.split(".").pop() || k] = v;
  return out;
}

function WorkspaceSkeleton() {
  return (
    <div className="rv" aria-busy="true" aria-label="Loading the review workspace">
      <div className="sheet-bar">
        <div className="row">
          <Skeleton width={40} height={40} radius={10} />
          <div className="stack-sm">
            <Skeleton width={160} height={20} />
            <Skeleton width={220} height={12} />
          </div>
        </div>
        <Skeleton width={260} height={36} radius={10} />
      </div>
      <Skeleton width="100%" height={48} radius={14} />
      <div className="workspace">
        <div className="split split--static">
          <div className="pane">
            <div className="viewer">
              <div className="paper paper--empty" style={{ width: "100%", aspectRatio: "1 / 1.2" }}>
                <span className="paper__loading skeleton" aria-hidden="true" />
              </div>
            </div>
          </div>
          <span />
          <div className="pane">
            <div className="ap">
              <Skeleton width={180} height={26} />
              <Skeleton lines={2} />
              <Skeleton width="100%" height={86} radius={10} />
              <Skeleton width="100%" height={150} radius={10} />
              <Skeleton lines={4} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ReviewWorkspace({ examId, submissionId }: { examId: string; submissionId: string }) {
  const navigate = useNavigate();
  const toast = useToast();
  const { user } = useAuth();
  const { exam, reload: reloadExam } = useExam();
  const caps = useCapabilities();
  const autoAccept = caps?.settings.auto_accept_threshold ?? 0.8;
  const contradictionThreshold = caps?.settings.contradiction_threshold ?? 0.5;
  const [params, setParams] = useSearchParams();

  const subQ = useAsync(() => api.getSubmission(submissionId), [submissionId]);
  const ansQ = useAsync(() => api.listAnswers(submissionId, { sort: "priority" }), [submissionId]);
  const pagesQ = useAsync(() => api.getPages(submissionId), [submissionId]);
  const regionsQ = useAsync(() => api.listRegions(submissionId), [submissionId]);
  const sheetsQ = useAsync(() => api.listAllSubmissions(examId), [examId]);

  const sub = subQ.data;
  useBreadcrumbs(2, [{ label: sub ? sub.student_ref : "Sheet" }]);

  // While the sheet is still being processed, poll and load everything once it is ready.
  const processing = !!sub && (sub.status === "uploaded" || sub.status === "processing");
  usePolling(
    async () => {
      const s = await api.getSubmission(submissionId);
      subQ.setData(s);
      if (s.status !== "uploaded" && s.status !== "processing") {
        void ansQ.reload(true);
        void pagesQ.reload(true);
        void regionsQ.reload(true);
        void reloadExam(true);
      }
    },
    3000,
    processing,
  );

  // ---------------------------------------------------------------- answers + selection
  const answers = useMemo(() => ansQ.data ?? [], [ansQ.data]);
  const byId = useMemo(() => new Map(answers.map((a) => [a.answer_id, a])), [answers]);
  const lookup = useMemo(() => questionLookup(answers.map((a) => ({ id: a.question_id, question_no: a.question_no }))), [answers]);
  const pendingAnswers = answers.filter(isPending);
  const pendingNos = pendingAnswers.map((a) => a.question_no).sort((x, y) => x - y);
  const [touched, setTouched] = useState<Set<string>>(() => new Set());
  const [filterState, setFilter] = useState<QueueFilter | null>(null);
  useEffect(() => {
    if (filterState === null && ansQ.data) setFilter(ansQ.data.some(isPending) ? "pending" : "all");
  }, [ansQ.data, filterState]);
  const filter: QueueFilter = filterState ?? "pending";
  const visible = useMemo(
    () => (filter === "all" ? answers : answers.filter((a) => isPending(a) || touched.has(a.answer_id))),
    [answers, filter, touched],
  );

  const paramId = params.get("answer");
  const currentId = paramId && byId.has(paramId) ? paramId : visible[0]?.answer_id ?? answers[0]?.answer_id ?? null;
  const current = currentId ? byId.get(currentId) ?? null : null;

  const select = useCallback(
    (id: string) =>
      setParams(
        (prev) => {
          const n = new URLSearchParams(prev);
          n.set("answer", id);
          return n;
        },
        { replace: true },
      ),
    [setParams],
  );
  // Pin the implicit selection into the URL so it stays stable as the queue changes.
  useEffect(() => {
    if (currentId && paramId !== currentId) select(currentId);
  }, [currentId, paramId, select]);

  const index = currentId ? visible.findIndex((a) => a.answer_id === currentId) : -1;
  const prevId = index > 0 ? visible[index - 1].answer_id : null;
  const nextId = index >= 0 ? visible[index + 1]?.answer_id ?? null : visible[0]?.answer_id ?? null;

  // ---------------------------------------------------------------- permissions
  // Only the exam's owner may change marks (an admin can edit their own exams, and only read others').
  const canEdit = canEditExam(exam);
  const isAdmin = user?.role === "admin";
  const sheetLocked = sub?.status === "approved";
  const sheetReviewable = sub?.status === "processed" || sub?.status === "in_review";
  const readOnly = !current || !sub || current.locked || sheetLocked || !canEdit || !sheetReviewable;

  // ---------------------------------------------------------------- draft (per answer)
  const [draftState, setDraftState] = useState<{ id: string; draft: Draft } | null>(null);
  const [errorState, setErrorState] = useState<{ id: string; errors: Record<string, string> } | null>(null);
  const draft: Draft | null = current ? (draftState?.id === current.answer_id ? draftState.draft : draftFrom(current)) : null;
  const errors = current && errorState?.id === current.answer_id ? errorState.errors : {};
  const status = current && draft ? draftStatus(current, draft) : null;
  const dirty = !!status?.dirty && !readOnly;

  const onDraft = useCallback(
    (patch: Partial<Draft>) => {
      if (!current) return;
      const id = current.answer_id;
      setDraftState((prev) => ({ id, draft: { ...(prev?.id === id ? prev.draft : draftFrom(current)), ...patch } }));
      setErrorState((prev) => {
        if (!prev || prev.id !== id) return prev;
        const next = { ...prev.errors };
        if ("text" in patch) delete next.corrected_text;
        if ("marks" in patch) {
          delete next.final_marks;
          delete next.reason;
        }
        if ("reason" in patch) delete next.reason;
        if ("comment" in patch) delete next.comment;
        return { id, errors: next };
      });
    },
    [current],
  );

  // ---------------------------------------------------------------- refs + UI state
  const textRef = useRef<HTMLTextAreaElement>(null);
  const marksRef = useRef<HTMLInputElement>(null);
  const reasonRef = useRef<HTMLTextAreaElement>(null);
  const approveSheetRef = useRef<HTMLButtonElement>(null);
  const panelBodyRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState<Busy>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [zoom, setZoom] = useLocalStorage<number>("saras-review-zoom", 1);
  const [pendingNav, setPendingNav] = useState<Nav | null>(null);
  const [confirmApprove, setConfirmApprove] = useState(false);
  const [confirmReopen, setConfirmReopen] = useState(false);
  const [sheetBusy, setSheetBusy] = useState<"approve" | "reopen" | null>(null);
  const [blocked, setBlocked] = useState<number[] | null>(null);
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    panelBodyRef.current?.scrollTo({ top: 0 });
    setStatusMsg("");
  }, [currentId]);

  // ---------------------------------------------------------------- navigation guard
  // While the draft is dirty every way off this page asks first: links, tabs, the sidebar,
  // sign-out and the browser's Back/Forward buttons.
  const { allow } = useUnsavedGuard(dirty, (resume, cancel) => setPendingNav({ kind: "resume", resume, cancel }));
  const go = useCallback(
    (nav: Nav) => {
      if (nav.kind === "answer") {
        // Every answer opens from its server state (no stale draft from an earlier visit).
        setDraftState(null);
        setErrorState(null);
        select(nav.id);
      } else if (nav.kind === "route") allow(() => navigate(nav.to)); // already confirmed
      else nav.resume();
    },
    [allow, navigate, select],
  );
  const request = (nav: Nav) => {
    if (nav.kind === "answer" && nav.id === currentId) return;
    if (dirty) setPendingNav(nav);
    else go(nav);
  };
  const discardAndGo = () => {
    const nav = pendingNav;
    setPendingNav(null);
    setDraftState(null);
    setErrorState(null);
    if (nav) go(nav);
  };
  const keepEditing = () => {
    if (pendingNav?.kind === "resume") pendingNav.cancel();
    setPendingNav(null);
  };

  // ---------------------------------------------------------------- helpers
  const applyAnswer = (u: AnswerOut) => {
    ansQ.setData((prev) => (prev ?? []).map((a) => (a.answer_id === u.answer_id ? u : a)));
    setTouched((prev) => new Set(prev).add(u.answer_id));
  };
  const refreshSheet = () => {
    void subQ.reload(true);
    void sheetsQ.reload(true);
    void reloadExam(true);
  };
  const focusFirstError = (errs: Record<string, string>) => {
    requestAnimationFrame(() => {
      if (errs.final_marks) marksRef.current?.focus();
      else if (errs.reason) reasonRef.current?.focus();
      else if (errs.corrected_text) textRef.current?.focus();
    });
  };
  const handleError = (err: unknown, title: string) => {
    if (err instanceof ApiError) {
      const fe = fieldErrorsOf(err);
      if ((err.status === 400 || err.status === 422) && Object.keys(fe).length && current) {
        setErrorState({ id: current.answer_id, errors: fe });
        focusFirstError(fe);
        toast.error(title, err.message);
        return;
      }
      if (err.status === 409) {
        toast.error(title, err.message);
        void subQ.reload(true);
        void ansQ.reload(true);
        return;
      }
      if (err.status === 403) {
        toast.error("You can view these marks but not change them", err.message);
        return;
      }
    }
    toast.error(title, errorMessage(err));
  };

  // ---------------------------------------------------------------- actions
  const approve = async () => {
    if (!current || !draft || readOnly || busy) return;
    const errs = validateDraft(current, draft, true);
    if (Object.keys(errs).length) {
      setErrorState({ id: current.answer_id, errors: errs });
      focusFirstError(errs);
      setStatusMsg("Fix the highlighted fields to approve.");
      return;
    }
    const st = draftStatus(current, draft);
    const body: ReviewIn = { approve: true };
    if (st.textChanged) body.corrected_text = draft.text;
    if (st.commentChanged) body.comment = draft.comment;
    if (st.marksChanged && draft.marks !== null) body.final_marks = draft.marks;
    if (draft.reason.trim()) body.reason = draft.reason.trim();
    setBusy("approve");
    try {
      const u = await api.reviewAnswer(current.answer_id, body);
      const list = answers.map((a) => (a.answer_id === u.answer_id ? u : a));
      applyAnswer(u);
      setDraftState(null);
      setErrorState(null);
      toast.success(
        `Question ${u.question_no} approved`,
        `${fmtMarks(u.final_marks)} / ${fmtMarks(u.max_marks)} marks${st.textChanged ? " · text corrected and re-graded" : ""}`,
      );
      refreshSheet();
      const at = list.findIndex((a) => a.answer_id === u.answer_id);
      const after = [...list.slice(at + 1), ...list.slice(0, at)];
      const next = after.find(isPending);
      if (next) select(next.answer_id);
      else {
        setStatusMsg("Every answer on this sheet is reviewed.");
        toast.info("Every answer is reviewed", "Approve the sheet to lock its marks.");
        requestAnimationFrame(() => approveSheetRef.current?.focus());
      }
    } catch (err) {
      handleError(err, "Could not approve the answer");
    } finally {
      setBusy(null);
    }
  };

  const saveText = async () => {
    if (!current || !draft || readOnly || busy) return;
    const st = draftStatus(current, draft);
    if (!st.textChanged && !st.commentChanged) {
      toast.info("Nothing to save", st.marksChanged ? "Marks are saved when you approve the answer." : undefined);
      return;
    }
    const errs = validateDraft(current, draft, false);
    if (Object.keys(errs).length) {
      setErrorState({ id: current.answer_id, errors: errs });
      focusFirstError(errs);
      return;
    }
    const body: ReviewIn = { approve: false };
    if (st.textChanged) body.corrected_text = draft.text;
    if (st.commentChanged) body.comment = draft.comment;
    setBusy("save");
    try {
      const u = await api.reviewAnswer(current.answer_id, body);
      applyAnswer(u);
      // Keep marks / reason the teacher typed; everything else now comes from the server.
      setDraftState(
        st.marksChanged || draft.reason.trim()
          ? { id: u.answer_id, draft: { ...draftFrom(u), marks: draft.marks, reason: draft.reason } }
          : null,
      );
      setErrorState(null);
      setStatusMsg("Saved.");
      toast.success(
        st.textChanged ? "Text saved and re-graded" : "Comment saved",
        st.textChanged ? `AI marks are now ${fmtMarks(u.ai_marks)} / ${fmtMarks(u.max_marks)}.` : undefined,
      );
      refreshSheet();
    } catch (err) {
      handleError(err, "Could not save");
    } finally {
      setBusy(null);
    }
  };

  const flag = async () => {
    if (!current || readOnly || busy) return;
    if (current.status === "needs_review" && current.flags.includes("teacher_flag")) {
      toast.info(`Question ${current.question_no} is already flagged`);
      return;
    }
    setBusy("flag");
    try {
      const u = await api.flagAnswer(current.answer_id, {});
      applyAnswer(u);
      setStatusMsg("Flagged for review.");
      toast.success(`Question ${u.question_no} flagged`, "It stays in the review queue until you approve it.");
      refreshSheet();
    } catch (err) {
      handleError(err, "Could not flag the answer");
    } finally {
      setBusy(null);
    }
  };

  const maxTotal = exam.rubric_marks || exam.total_marks;

  const approveSheet = async () => {
    if (!sub) return;
    setSheetBusy("approve");
    try {
      const s = await api.approveSubmission(sub.id);
      subQ.setData(s);
      setConfirmApprove(false);
      setBlocked(null);
      setDraftState(null);
      setErrorState(null);
      toast.success(`Sheet ${s.student_ref} approved`, `Final total ${fmtMarks(s.total_final_marks)} / ${fmtMarks(maxTotal)}. Marks are locked and included in the export.`);
      void ansQ.reload(true);
      void sheetsQ.reload(true);
      void reloadExam(true);
    } catch (err) {
      setConfirmApprove(false);
      if (err instanceof ApiError && err.status === 409) {
        const pend = (err.details as { pending?: unknown } | null | undefined)?.pending;
        if (Array.isArray(pend) && pend.length) {
          setBlocked(pend.map(Number));
          toast.error("Some answers still need review", `Questions ${pend.join(", ")}`);
        } else toast.error("Could not approve the sheet", err.message);
        void ansQ.reload(true);
        void subQ.reload(true);
      } else toast.error("Could not approve the sheet", errorMessage(err));
    } finally {
      setSheetBusy(null);
    }
  };

  const reopen = async (reason: string) => {
    if (!sub) return;
    setSheetBusy("reopen");
    try {
      const s = await api.reopenSubmission(sub.id, reason);
      subQ.setData(s);
      setConfirmReopen(false);
      toast.success(`Sheet ${s.student_ref} reopened`, "You can change marks again. Approve it when you're done.");
      void ansQ.reload(true);
      void sheetsQ.reload(true);
      void reloadExam(true);
    } catch (err) {
      toast.error("Could not reopen the sheet", errorMessage(err));
    } finally {
      setSheetBusy(null);
    }
  };

  // ---------------------------------------------------------------- sheets (picker / next)
  const sheets = useMemo(
    () => (sheetsQ.data ?? []).filter((s) => s.status !== "uploaded" && s.status !== "processing" && s.status !== "failed").sort(byRef),
    [sheetsQ.data],
  );
  const nextSheet = useMemo(() => {
    const at = sheets.findIndex((s) => s.id === submissionId);
    const after = [...sheets.slice(at + 1), ...sheets.slice(0, Math.max(0, at))].filter((s) => s.id !== submissionId && s.status !== "approved");
    return after.find((s) => s.answers_needing_review > 0) ?? after[0] ?? null;
  }, [sheets, submissionId]);
  const sheetHref = (id: string) => `/exams/${examId}/review/${id}`;
  const regionsHref = `/exams/${examId}/regions/${submissionId}`;

  // ---------------------------------------------------------------- keyboard
  useHotkeys((e) => {
    if (modalOpen() || pendingNav) return;
    const k = e.key;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && k === "Enter") {
      e.preventDefault();
      void approve();
      return;
    }
    if (mod && (k === "s" || k === "S")) {
      e.preventDefault();
      void saveText();
      return;
    }
    if (mod || e.altKey || isTypingTarget(e.target)) return;
    if (k.startsWith("Arrow") && inArrowWidget(e.target)) return;
    switch (k) {
      case "j":
      case "J":
      case "ArrowDown":
        if (nextId) {
          e.preventDefault();
          request({ kind: "answer", id: nextId });
        }
        break;
      case "k":
      case "K":
      case "ArrowUp":
        if (prevId) {
          e.preventDefault();
          request({ kind: "answer", id: prevId });
        }
        break;
      case "a":
      case "A":
        if (!readOnly) {
          e.preventDefault();
          void approve();
        }
        break;
      case "f":
      case "F":
        if (!readOnly) {
          e.preventDefault();
          void flag();
        }
        break;
      case "e":
      case "E":
        e.preventDefault();
        textRef.current?.focus();
        break;
      case "?":
        e.preventDefault();
        setHelpOpen((o) => !o);
        break;
      case "+":
      case "=":
        e.preventDefault();
        setZoom(zoomStep(zoom, 1));
        break;
      case "-":
      case "_":
        e.preventDefault();
        setZoom(zoomStep(zoom, -1));
        break;
      case "0":
        e.preventDefault();
        setZoom(1);
        break;
    }
  }, !!sub);

  // ---------------------------------------------------------------- render
  if (subQ.error && !sub) {
    const missing = subQ.error instanceof ApiError && (subQ.error.status === 404 || subQ.error.status === 403);
    return missing ? (
      <Card>
        <EmptyState
          illustration="review"
          title="Sheet not found"
          description="It may have been deleted, or it belongs to another exam."
          action={
            <ButtonLink to={`/exams/${examId}/review`} variant="primary">
              Back to the review queue
            </ButtonLink>
          }
        />
      </Card>
    ) : (
      <ErrorBanner error={subQ.error} onRetry={() => subQ.reload()} title="Could not load this sheet" />
    );
  }
  if (!sub || (ansQ.loading && !ansQ.data)) return <WorkspaceSkeleton />;

  const shortcutsNode = (
    <ShortcutHelp
      shortcuts={SHORTCUTS}
      open={helpOpen}
      onOpenChange={setHelpOpen}
      note="Shortcuts pause while you type in a field, except Ctrl + Enter and Ctrl + S."
    />
  );

  const header = (
    <SheetHeader
      sub={sub}
      sheets={sheets}
      maxTotal={maxTotal}
      pendingNos={pendingNos}
      canEdit={canEdit}
      sheetBusy={sheetBusy}
      approveRef={approveSheetRef}
      nextSheet={nextSheet}
      shortcuts={shortcutsNode}
      onBack={() => request({ kind: "route", to: `/exams/${examId}/review` })}
      onPickSheet={(id) => id !== submissionId && request({ kind: "route", to: sheetHref(id) })}
      onApprove={() => setConfirmApprove(true)}
      onJumpPending={() => {
        const first = pendingAnswers[0];
        toast.info("Some answers still need review", `Approve ${pendingNos.map((n) => `Q${n}`).join(", ")} first.`);
        if (first) request({ kind: "answer", id: first.answer_id });
      }}
      onReopen={() => setConfirmReopen(true)}
      onNextSheet={() => nextSheet && request({ kind: "route", to: sheetHref(nextSheet.id) })}
    />
  );

  // Sheet not ready / failed: no workspace yet.
  if (processing || sub.status === "failed") {
    return (
      <div className="rv">
        {header}
        <Card>
          {processing ? (
            <EmptyState
              illustration="review"
              title="This sheet is still being processed"
              description={sub.stage ? `Current step: ${stageLabel(sub).toLowerCase()}. The review opens automatically when it's ready.` : "The review opens automatically when it's ready."}
              action={
                <div className="rv__progress">
                  <ProgressBar
                    value={sub.page_count ? sub.pages_done / sub.page_count : null}
                    label="Pages processed"
                    showLabel
                    valueText={sub.page_count ? `${sub.pages_done} of ${sub.page_count} pages` : undefined}
                  />
                </div>
              }
            />
          ) : (
            <EmptyState
              illustration="review"
              title="Processing failed for this sheet"
              description={sub.error || "The sheet could not be read. Retry it from the Status tab."}
              action={
                <ButtonLink to={`/exams/${examId}/status`} variant="primary">
                  Go to Status
                </ButtonLink>
              }
            />
          )}
        </Card>
      </div>
    );
  }

  const position = index >= 0 ? `${index + 1} of ${visible.length}` : `${visible.length} listed`;
  const actionStatus = readOnly
    ? sheetLocked
      ? "Approved · read only"
      : !canEdit && isAdmin
        ? "Admin view · read only"
        : "Read only"
    : dirty
      ? "Unsaved changes"
      : statusMsg;

  return (
    <div className="rv">
      {header}

      {blocked && !sheetLocked ? (
        <Banner
          tone="warning"
          title="The sheet can't be approved yet"
          action={
            <Button size="sm" variant="secondary" onClick={() => setBlocked(null)}>
              Dismiss
            </Button>
          }
        >
          Still needing review: {blocked.map((n) => `Q${n}`).join(", ")}.
        </Banner>
      ) : null}
      {sheetLocked ? (
        <Banner tone="success" title={`Approved${sub.approved_at ? ` on ${formatDate(sub.approved_at)}` : ""}`}>
          Marks on this sheet are locked and included in the CSV export.{canEdit ? " Reopen the sheet to change them." : ""}
        </Banner>
      ) : !canEdit ? (
        <Banner tone="info" title={isAdmin ? "Read-only admin view" : "Read-only view"}>
          {isAdmin ? "Admins can read every exam but only change marks on their own exams." : "Only the exam's owner can change these marks."}
        </Banner>
      ) : null}
      {ansQ.error ? <ErrorBanner error={ansQ.error} onRetry={() => ansQ.reload()} title="Could not refresh the answers" /> : null}

      {answers.length === 0 ? (
        <Card>
          <EmptyState
            illustration="review"
            title="No answers on this sheet"
            description="Nothing was graded for this sheet. Check the rubric, or the answer regions."
            action={
              <ButtonLink to={regionsHref} variant="secondary">
                Open the region editor
              </ButtonLink>
            }
          />
        </Card>
      ) : (
        <>
          <AnswerQueue
            visible={visible}
            total={answers.length}
            pending={pendingAnswers.length}
            filter={filter}
            onFilter={setFilter}
            currentId={currentId}
            onSelect={(id) => request({ kind: "answer", id })}
            refreshing={ansQ.fetching && !!ansQ.data}
          />

          <div className="workspace">
            <SplitPane
              storageKey="saras-review-split"
              initial={0.5}
              min={0.32}
              max={0.68}
              label="Resize the scan and answer panels"
              left={
                <ScanViewer
                  submissionId={submissionId}
                  pages={pagesQ.data?.pages}
                  pagesError={pagesQ.error}
                  onRetry={() => {
                    void pagesQ.reload();
                    void regionsQ.reload();
                  }}
                  regions={regionsQ.data}
                  answer={current}
                  lookup={lookup}
                  zoom={zoom}
                  onZoom={setZoom}
                  regionsHref={regionsHref}
                  canEditRegions={canEdit && !sheetLocked}
                />
              }
              right={
                <section className="pane pane--answer" aria-label="Answer review">
                  <div className="pane__body" ref={panelBodyRef}>
                    {current && draft ? (
                      <AnswerPanel
                        key={current.answer_id}
                        answer={current}
                        draft={draft}
                        onDraft={onDraft}
                        errors={errors}
                        readOnly={readOnly}
                        autoAccept={autoAccept}
                        contradictionThreshold={contradictionThreshold}
                        regionsHref={regionsHref}
                        textRef={textRef}
                        marksRef={marksRef}
                        reasonRef={reasonRef}
                      />
                    ) : (
                      <EmptyState compact illustration="review" title="Pick an answer" description="Choose a question from the queue above." />
                    )}
                  </div>
                  <div className="pane__footer actionbar">
                    <div className="actionbar__nav">
                      <IconButton
                        size="sm"
                        variant="secondary"
                        label="Previous answer (K)"
                        icon={<ChevronUp size={16} aria-hidden="true" />}
                        disabled={!prevId}
                        aria-keyshortcuts="K ArrowUp"
                        onClick={() => prevId && request({ kind: "answer", id: prevId })}
                      />
                      <span className="actionbar__pos num">{position}</span>
                      <IconButton
                        size="sm"
                        variant="secondary"
                        label="Next answer (J)"
                        icon={<ChevronDown size={16} aria-hidden="true" />}
                        disabled={!nextId}
                        aria-keyshortcuts="J ArrowDown"
                        onClick={() => nextId && request({ kind: "answer", id: nextId })}
                      />
                    </div>
                    <p className={cx("actionbar__status", dirty && "is-dirty")} role="status" aria-live="polite">
                      {actionStatus}
                    </p>
                    {!readOnly ? (
                      <div className="actionbar__main">
                        <Button
                          size="sm"
                          variant="ghost"
                          iconLeft={<Flag size={14} aria-hidden="true" />}
                          loading={busy === "flag"}
                          disabled={!!busy && busy !== "flag"}
                          aria-keyshortcuts="F"
                          title="Keep this answer in the review queue (F)"
                          onClick={() => void flag()}
                        >
                          Flag <Kbd>F</Kbd>
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          iconLeft={<Save size={14} aria-hidden="true" />}
                          loading={busy === "save"}
                          disabled={(!!busy && busy !== "save") || !(status?.textChanged || status?.commentChanged)}
                          aria-keyshortcuts="Control+S"
                          title="Save the corrected text and comment without approving (Ctrl+S)"
                          onClick={() => void saveText()}
                        >
                          Save text
                        </Button>
                        <Button
                          size="sm"
                          variant="primary"
                          iconLeft={<Check size={14} aria-hidden="true" />}
                          loading={busy === "approve"}
                          disabled={!!busy && busy !== "approve"}
                          aria-keyshortcuts="A Control+Enter"
                          title="Approve these marks and go to the next pending answer (A or Ctrl+Enter)"
                          onClick={() => void approve()}
                        >
                          {current?.status === "reviewed" && !dirty ? "Re-approve" : "Approve answer"} <Kbd>A</Kbd>
                        </Button>
                      </div>
                    ) : null}
                  </div>
                </section>
              }
            />
          </div>
        </>
      )}

      <ConfirmDialog
        open={!!pendingNav}
        tone="danger"
        title="Discard unsaved changes?"
        message={
          <>
            Your edits to <strong>Question {current?.question_no}</strong> have not been saved. Leave without saving?
          </>
        }
        confirmLabel="Discard changes"
        cancelLabel="Keep editing"
        onConfirm={discardAndGo}
        onCancel={keepEditing}
      />
      <ConfirmDialog
        open={confirmApprove}
        title={`Approve sheet ${sub.student_ref}?`}
        message={
          <>
            <p>
              Final total <strong className="num">{fmtMarks(sub.total_final_marks)}</strong> / {fmtMarks(maxTotal)}. The marks will be locked and
              included in the CSV export.
            </p>
            {dirty ? <p className="text-warning">Unsaved edits to Question {current?.question_no} will be discarded.</p> : null}
          </>
        }
        confirmLabel="Approve sheet"
        busy={sheetBusy === "approve"}
        onConfirm={() => void approveSheet()}
        onCancel={() => setConfirmApprove(false)}
      />
      <ConfirmDialog
        open={confirmReopen}
        title="Reopen this sheet?"
        message="Its marks become editable again and it leaves the export until you approve it again."
        reasonLabel="Reason for reopening"
        reasonRequired
        confirmLabel="Reopen sheet"
        busy={sheetBusy === "reopen"}
        onConfirm={(reason) => void reopen(reason)}
        onCancel={() => setConfirmReopen(false)}
      />
    </div>
  );
}
