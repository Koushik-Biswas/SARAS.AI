// Region editor (PRD 17.1 screen 7): page thumbnails, an interactive page canvas with every
// answer region, and an inspector to reassign / crop / split / merge / blank / draw / undo.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, ChevronLeft, ChevronRight, Maximize, MousePointer2, SquarePlus, Undo2 } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { AnswerOut, BBox, PageInfo, RegionEditOut, RegionOut } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import {
  Badge,
  Banner,
  Button,
  ButtonLink,
  Card,
  DelayedSpinner,
  EmptyState,
  ErrorBanner,
  Field,
  IconButton,
  Kbd,
  Modal,
  Segmented,
  Select,
  Skeleton,
  StatusPill,
  cx,
  useToast,
} from "../components";
import { useAsync } from "../hooks/useAsync";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { useBreadcrumbs } from "../layout/ShellContext";
import { canEditExam, useExam } from "../pages/ExamLayout";
import { ShortcutHelp, ZOOM_STEPS, ZoomControls, zoomStep, type ShortcutDef } from "./controls";
import { flagSpec } from "./flags";
import { inArrowWidget, isTypingTarget, modalOpen, useHotkeys } from "./hooks";
import { Paper } from "./Paper";
import { RegionCanvas, type EditorMode } from "./RegionCanvas";
import { RegionInspector, type ChangeSummary } from "./RegionInspector";
import { boxStyle, questionLookup, regionStyle, type QuestionLookup } from "./regions";

const SHORTCUTS: ShortcutDef[] = [
  { label: "Select / move / resize", keys: [["V"]] },
  { label: "Draw a new region", keys: [["N"]] },
  { label: "Split the selected region", keys: [["S"]] },
  { label: "Merge with another region", keys: [["M"]] },
  { label: "Mark blank / answered", keys: [["B"]] },
  { label: "Undo the last edit", keys: [["Ctrl", "Z"]] },
  { label: "Previous / next page", keys: [["["], ["]"]] },
  { label: "Zoom in / out / fit width", keys: [["+"], ["−"], ["0"]] },
  { label: "Cancel / deselect", keys: [["Esc"]] },
  { label: "Show this help", keys: [["?"]] },
];

const MODE_HINT: Record<EditorMode, string> = {
  select: "",
  draw: "Drag on the page to draw a new answer region.",
  split: "Click inside the selected box where it should be split in two.",
  merge: "Click another region to merge it into the selected one.",
};

const ACTION_LABEL: Record<string, string> = {
  crop: "Region resized",
  reassign: "Region reassigned",
  split: "Region split in two",
  merge: "Regions merged",
  blank: "Marked blank",
  unblank: "Marked as answered",
  create: "Region added",
  undo: "Last edit undone",
};

// ---------------------------------------------------------------- thumbnails
function PageThumbs({
  submissionId,
  pages,
  regionsByPage,
  lookup,
  current,
  onPick,
}: {
  submissionId: string;
  pages: PageInfo[];
  regionsByPage: Map<number, RegionOut[]>;
  lookup: QuestionLookup;
  current: number;
  onPick: (n: number) => void;
}) {
  return (
    <nav className="thumbs" aria-label="Pages">
      <ol className="thumbs__list">
        {pages.map((p) => {
          const rs = regionsByPage.get(p.page_no) ?? [];
          const active = p.page_no === current;
          const flagText = p.flags.map((f) => flagSpec(f).label).join(", ");
          return (
            <li key={p.page_no}>
              <button
                type="button"
                className={cx("thumb", active && "is-active")}
                aria-current={active ? "page" : undefined}
                aria-label={`Page ${p.page_no}, ${rs.length} region${rs.length === 1 ? "" : "s"}${flagText ? `, ${flagText}` : ""}`}
                onClick={() => onPick(p.page_no)}
              >
                <Paper submissionId={submissionId} page={p} thumb alt="">
                  {(space) =>
                    rs.map((r) => (
                      <span
                        key={r.id}
                        className={cx("tbox", !r.question_id && "tbox--dashed")}
                        style={{ ...boxStyle(r.bbox, space.w, space.h), ["--rc" as string]: regionStyle(r, lookup).color }}
                      />
                    ))
                  }
                </Paper>
                <span className="thumb__meta">
                  <span className="thumb__no">Page {p.page_no}</span>
                  <span className="thumb__count num" aria-hidden="true">
                    {rs.length}
                  </span>
                </span>
                {p.flags.length ? (
                  <span className="thumb__flags" aria-hidden="true">
                    {p.flags.map((f) => {
                      const spec = flagSpec(f);
                      const Icon = spec.icon;
                      return (
                        <Badge key={f} size="sm" tone={spec.tone} icon={<Icon size={11} />}>
                          {spec.label}
                        </Badge>
                      );
                    })}
                  </span>
                ) : null}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function EditorSkeleton() {
  return (
    <div className="re" aria-busy="true" aria-label="Loading the region editor">
      <Skeleton width="100%" height={52} radius={14} />
      <div className="re-grid">
        <div className="thumbs">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} width="100%" height={150} radius={10} />
          ))}
        </div>
        <div className="rcanvas">
          <div className="paper paper--empty" style={{ width: "100%", aspectRatio: "1 / 1.3" }}>
            <span className="paper__loading skeleton" aria-hidden="true" />
          </div>
        </div>
        <div className="insp">
          <Skeleton lines={3} />
          <Skeleton width="100%" height={120} radius={10} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- editor
export function RegionEditor({ examId, submissionId }: { examId: string; submissionId: string }) {
  const toast = useToast();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { exam, reload: reloadExam } = useExam();
  const [params, setParams] = useSearchParams();

  const subQ = useAsync(() => api.getSubmission(submissionId), [submissionId]);
  const pagesQ = useAsync(() => api.getPages(submissionId), [submissionId]);
  const regionsQ = useAsync(() => api.listRegions(submissionId), [submissionId]);
  const questionsQ = useAsync(() => api.getQuestions(examId), [examId]);
  const answersQ = useAsync(() => api.listAnswers(submissionId, { sort: "question" }), [submissionId]);

  const sub = subQ.data;
  const reviewHref = useCallback(
    (answerId?: string) => `/exams/${examId}/review/${submissionId}${answerId ? `?answer=${encodeURIComponent(answerId)}` : ""}`,
    [examId, submissionId],
  );
  useBreadcrumbs(2, [{ label: sub ? sub.student_ref : "Sheet", to: reviewHref() }, { label: "Answer regions" }]);

  const pages = useMemo(() => pagesQ.data?.pages ?? [], [pagesQ.data]);
  const regions = useMemo(() => regionsQ.data ?? [], [regionsQ.data]);
  const questions = useMemo(() => questionsQ.data ?? [], [questionsQ.data]);
  const lookup = useMemo(() => questionLookup(questions), [questions]);
  const answersByQuestion = useMemo(() => new Map((answersQ.data ?? []).map((a) => [a.question_id, a])), [answersQ.data]);
  const regionsByPage = useMemo(() => {
    const m = new Map<number, RegionOut[]>();
    for (const r of regions) m.set(r.page_no, [...(m.get(r.page_no) ?? []), r]);
    return m;
  }, [regions]);

  // ---------------------------------------------------------------- page + selection
  const pageParam = Number(params.get("page"));
  const firstWithRegions = pages.find((p) => regionsByPage.has(p.page_no))?.page_no ?? pages[0]?.page_no ?? 1;
  const pageNo = pages.some((p) => p.page_no === pageParam) ? pageParam : firstWithRegions;
  const page = pages.find((p) => p.page_no === pageNo) ?? null;
  const setPage = useCallback(
    (n: number) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("page", String(n));
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = regions.find((r) => r.id === selectedId) ?? null;
  const [mode, setMode] = useState<EditorMode>("select");
  const [busy, setBusy] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [splitY, setSplitY] = useState<number | null>(null);
  const [draftBox, setDraftBox] = useState<BBox | null>(null);
  const [draftQ, setDraftQ] = useState("");
  const [lastChange, setLastChange] = useState<ChangeSummary | null>(null);
  const [zoom, setZoom] = useLocalStorage<number>("saras-regions-zoom", 1);
  const [helpOpen, setHelpOpen] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Only the exam's owner edits regions (an admin can edit their own exams, and only read others').
  const canEdit = canEditExam(exam);
  const isAdmin = user?.role === "admin";
  const editableStatus = sub?.status === "processed" || sub?.status === "in_review";
  const readOnly = !sub || !editableStatus || !canEdit;

  // Selection follows the page (except while picking a merge partner on another page);
  // split / merge need a selection.
  const prevPage = useRef(pageNo);
  useEffect(() => {
    if (prevPage.current === pageNo) return;
    prevPage.current = pageNo;
    if (selected && selected.page_no !== pageNo && mode !== "merge") setSelectedId(null);
  }, [pageNo, selected, mode]);
  useEffect(() => {
    if (!selected && (mode === "split" || mode === "merge")) setMode("select");
  }, [selected, mode]);
  useEffect(() => setSplitY(null), [selectedId]);
  useEffect(() => {
    canvasRef.current?.scrollTo({ top: 0, left: 0 });
  }, [pageNo]);

  // ---------------------------------------------------------------- server actions
  const applyEdit = (res: RegionEditOut, action: string) => {
    const before = new Map((answersQ.data ?? []).map((a) => [a.answer_id, a.final_marks ?? a.ai_marks]));
    regionsQ.setData(res.regions);
    answersQ.setData((prev) => {
      const map = new Map(res.answers.map((a) => [a.answer_id, a]));
      const list = (prev ?? []).map((a) => map.get(a.answer_id) ?? a);
      for (const a of res.answers) if (!list.some((x) => x.answer_id === a.answer_id)) list.push(a);
      return list.sort((x, y) => x.question_no - y.question_no);
    });
    setSelectedId((cur) => res.region?.id ?? (cur && res.regions.some((r) => r.id === cur) ? cur : null));
    const label = ACTION_LABEL[action] ?? "Regions updated";
    setLastChange({ label, rows: res.answers.map((a: AnswerOut) => ({ answer: a, before: before.get(a.answer_id) ?? null })) });
    setReason("");
    setSplitY(null);
    setMode("select");
    toast.success(label, res.answers.length ? `Re-graded ${res.answers.map((a) => `Q${a.question_no}`).join(", ")}.` : undefined);
    void subQ.reload(true);
    void reloadExam(true);
  };

  const handleError = (err: unknown, action: string) => {
    if (err instanceof ApiError) {
      if (action === "undo" && err.status === 409) {
        toast.info("Nothing to undo", err.message === "Nothing to undo" ? undefined : err.message);
        return;
      }
      if (err.status === 400 || err.status === 422) {
        const detail = Object.values(err.fieldErrors).join("; ");
        toast.error(err.message, detail || undefined);
        return;
      }
      if (err.status === 409) {
        toast.error("The sheet changed", err.message);
        void subQ.reload(true);
        void regionsQ.reload(true);
        return;
      }
      if (err.status === 403) {
        toast.error("You can view regions but not change them", err.message);
        return;
      }
    }
    toast.error("Could not save the change", errorMessage(err));
  };

  const run = async (action: string, fn: () => Promise<RegionEditOut>): Promise<boolean> => {
    if (busy || readOnly) return false;
    setBusy(action);
    try {
      applyEdit(await fn(), action);
      return true;
    } catch (err) {
      handleError(err, action);
      return false;
    } finally {
      setBusy(null);
    }
  };

  const why = () => reason.trim() || undefined;
  const reassign = (id: string, qid: string | null) => run("reassign", () => api.patchRegion(id, { action: "reassign", question_id: qid, reason: why() }));
  const crop = (id: string, bbox: BBox) => run("crop", () => api.patchRegion(id, { action: "crop", bbox, reason: why() }));
  const split = (id: string, y: number) => run("split", () => api.patchRegion(id, { action: "split", split_y: Math.round(y), reason: why() }));
  const merge = (id: string, other: string) => run("merge", () => api.patchRegion(id, { action: "merge", merge_with: other, reason: why() }));
  const blank = (id: string, value: boolean) =>
    run(value ? "blank" : "unblank", () => api.patchRegion(id, { action: value ? "blank" : "unblank", reason: why() }));
  const undo = () => run("undo", () => api.undoRegions(submissionId));

  const onDrawn = (bbox: BBox) => {
    const used = new Set(regions.map((r) => r.question_id).filter(Boolean));
    const sorted = [...questions].sort((a, b) => a.question_no - b.question_no);
    const guess = selected?.question_id ?? sorted.find((q) => !used.has(q.id))?.id ?? sorted[0]?.id ?? "";
    setDraftQ(guess);
    setDraftBox(bbox);
  };
  const confirmDraft = async () => {
    if (!draftBox || !page) return;
    const ok = await run("create", () => api.createRegion(submissionId, { page_no: page.page_no, bbox: draftBox, question_id: draftQ || null }));
    if (ok) setDraftBox(null);
  };

  const pickQuestion = (qid: string | null) => {
    const r = regions.find((x) => x.question_id === qid);
    if (!r) return;
    setMode("select");
    setPage(r.page_no);
    setSelectedId(r.id);
  };

  const pageIndex = pages.findIndex((p) => p.page_no === pageNo);
  const stepPage = (dir: 1 | -1) => {
    const p = pages[pageIndex + dir];
    if (p) setPage(p.page_no);
  };

  const fitPage = () => {
    const el = canvasRef.current;
    if (!el || !page || !page.width || !page.height) return;
    const cs = getComputedStyle(el);
    const availW = el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const availH = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom) - 4;
    if (availW <= 0 || availH <= 0) return;
    const z = availH / ((availW * page.height) / page.width);
    setZoom(Math.max(ZOOM_STEPS[0], Math.min(1, Math.round(z * 100) / 100)));
  };

  // ---------------------------------------------------------------- keyboard
  useHotkeys((e) => {
    if (modalOpen()) return;
    const k = e.key;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (k === "z" || k === "Z") && !e.shiftKey && !isTypingTarget(e.target)) {
      e.preventDefault();
      void undo();
      return;
    }
    if (k === "Escape") {
      if (helpOpen) return;
      if (mode !== "select") setMode("select");
      else if (selectedId && !isTypingTarget(e.target)) setSelectedId(null);
      return;
    }
    if (mod || e.altKey || isTypingTarget(e.target)) return;
    if (k.startsWith("Arrow") && inArrowWidget(e.target)) return;
    switch (k) {
      case "v":
      case "V":
        setMode("select");
        break;
      case "n":
      case "N":
        if (!readOnly) setMode((m) => (m === "draw" ? "select" : "draw"));
        break;
      case "s":
      case "S":
        if (!readOnly && selected) setMode((m) => (m === "split" ? "select" : "split"));
        break;
      case "m":
      case "M":
        if (!readOnly && selected) setMode((m) => (m === "merge" ? "select" : "merge"));
        break;
      case "b":
      case "B":
        if (!readOnly && selected) void blank(selected.id, !selected.is_blank);
        break;
      case "[":
      case "PageUp":
        e.preventDefault();
        stepPage(-1);
        break;
      case "]":
      case "PageDown":
        e.preventDefault();
        stepPage(1);
        break;
      case "+":
      case "=":
        setZoom(zoomStep(zoom, 1));
        break;
      case "-":
      case "_":
        setZoom(zoomStep(zoom, -1));
        break;
      case "0":
        setZoom(1);
        break;
      case "?":
        setHelpOpen((o) => !o);
        break;
      default:
        return;
    }
    e.preventDefault();
  }, !!sub);

  // ---------------------------------------------------------------- render
  const loadError = subQ.error || pagesQ.error || regionsQ.error || questionsQ.error;
  if (loadError && !(sub && pagesQ.data && regionsQ.data && questionsQ.data)) {
    const missing = loadError instanceof ApiError && (loadError.status === 404 || loadError.status === 403);
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
      <ErrorBanner
        error={loadError}
        title="Could not load the region editor"
        onRetry={() => {
          void subQ.reload();
          void pagesQ.reload();
          void regionsQ.reload();
          void questionsQ.reload();
        }}
      />
    );
  }
  if (!sub || !pagesQ.data || !regionsQ.data || !questionsQ.data) return <EditorSkeleton />;

  const processing = sub.status === "uploaded" || sub.status === "processing";

  return (
    <div className="re">
      <div className="re-toolbar" role="group" aria-label="Region editor tools">
        <Button size="sm" variant="ghost" iconLeft={<ArrowLeft size={14} aria-hidden="true" />} onClick={() => navigate(reviewHref())}>
          Back to review
        </Button>
        <span className="re-toolbar__sep" aria-hidden="true" />
        <div className="re-toolbar__sheet">
          <strong className="truncate">{sub.student_ref}</strong>
          <StatusPill kind="submission" status={sub.status} size="sm" />
        </div>
        <span className="re-toolbar__sep" aria-hidden="true" />
        <div className="re-toolbar__pages" role="group" aria-label="Page">
          <IconButton size="sm" label="Previous page ([)" icon={<ChevronLeft size={16} aria-hidden="true" />} disabled={pageIndex <= 0} onClick={() => stepPage(-1)} />
          <span className="num re-toolbar__page">
            Page {pageNo} <span className="subtle">of {pages.length}</span>
          </span>
          <IconButton
            size="sm"
            label="Next page (])"
            icon={<ChevronRight size={16} aria-hidden="true" />}
            disabled={pageIndex < 0 || pageIndex >= pages.length - 1}
            onClick={() => stepPage(1)}
          />
        </div>
        <div className="re-toolbar__grow" />
        <DelayedSpinner loading={!!busy} size={14} label="Saving and re-grading" />
        {!readOnly ? (
          <Segmented<"select" | "draw">
            size="sm"
            label="Tool"
            value={mode === "draw" ? "draw" : "select"}
            onChange={(v) => setMode(v)}
            options={[
              { value: "select", label: "Select", icon: <MousePointer2 size={14} aria-hidden="true" /> },
              { value: "draw", label: "Draw", icon: <SquarePlus size={14} aria-hidden="true" /> },
            ]}
          />
        ) : null}
        <ZoomControls
          zoom={zoom}
          onZoom={setZoom}
          extra={<IconButton size="sm" label="Fit whole page" icon={<Maximize size={15} aria-hidden="true" />} onClick={fitPage} />}
        />
        {!readOnly ? (
          <Button size="sm" variant="secondary" iconLeft={<Undo2 size={14} aria-hidden="true" />} loading={busy === "undo"} disabled={!!busy && busy !== "undo"} onClick={() => void undo()} title="Undo the last region edit (Ctrl+Z)">
            Undo <Kbd>Ctrl Z</Kbd>
          </Button>
        ) : null}
        <ShortcutHelp shortcuts={SHORTCUTS} open={helpOpen} onOpenChange={setHelpOpen} note="Every edit re-grades the affected answers right away." />
      </div>

      {sub.status === "approved" ? (
        <Banner tone="success" title="This sheet is approved">
          Regions are read only. Reopen the sheet from the review screen to change them.
        </Banner>
      ) : !canEdit ? (
        <Banner tone="info" title={isAdmin ? "Read-only admin view" : "Read-only view"}>
          {isAdmin ? "Admins can read every exam but only change regions and marks on their own exams." : "Only the exam's owner can change regions or marks."}
        </Banner>
      ) : processing ? (
        <Banner tone="info" title="This sheet is still being processed">
          Regions appear when processing finishes.
        </Banner>
      ) : sub.status === "failed" ? (
        <Banner tone="danger" title="Processing failed for this sheet">
          {sub.error || "Retry it from the Status tab."}
        </Banner>
      ) : null}

      {pages.length === 0 ? (
        <Card>
          <EmptyState illustration="generic" title="No pages yet" description="The sheet has no rendered pages to edit." />
        </Card>
      ) : (
        <div className="re-grid">
          <PageThumbs submissionId={submissionId} pages={pages} regionsByPage={regionsByPage} lookup={lookup} current={pageNo} onPick={setPage} />

          <div className={cx("rcanvas", `rcanvas--mode-${mode}`)} ref={canvasRef}>
            {MODE_HINT[mode] && !readOnly ? (
              <div className="rcanvas__hint" role="status">
                <span>{MODE_HINT[mode]}</span>
                <Button size="sm" variant="ghost" onClick={() => setMode("select")}>
                  Cancel <Kbd>Esc</Kbd>
                </Button>
              </div>
            ) : null}
            {page ? (
              <RegionCanvas
                key={page.page_no}
                submissionId={submissionId}
                page={page}
                regions={regionsByPage.get(page.page_no) ?? []}
                lookup={lookup}
                selectedId={selectedId}
                mode={mode}
                readOnly={readOnly}
                busy={!!busy}
                zoom={zoom}
                draftBox={draftBox}
                splitY={splitY}
                onSelect={setSelectedId}
                onCancelMode={() => setMode("select")}
                onCommitBox={(id, bbox) => crop(id, bbox)}
                onSplit={(id, y) => void split(id, y)}
                onDrawn={onDrawn}
                onMergePick={(other) => selected && void merge(selected.id, other)}
              />
            ) : null}
          </div>

          <RegionInspector
            region={selected}
            regions={regions}
            questions={questions}
            lookup={lookup}
            page={selected ? pages.find((p) => p.page_no === selected.page_no) ?? null : page}
            mode={mode}
            readOnly={readOnly}
            busy={busy}
            reason={reason}
            splitY={splitY}
            answersByQuestion={answersByQuestion}
            lastChange={lastChange}
            reviewHref={reviewHref}
            onReason={setReason}
            onSplitY={setSplitY}
            onMode={setMode}
            onDeselect={() => setSelectedId(null)}
            onReassign={(qid) => selected && void reassign(selected.id, qid)}
            onCrop={(bbox) => selected && void crop(selected.id, bbox)}
            onSplit={(y) => selected && void split(selected.id, y)}
            onMerge={(other) => selected && void merge(selected.id, other)}
            onBlank={(v) => selected && void blank(selected.id, v)}
            onPickQuestion={pickQuestion}
          />
        </div>
      )}

      <Modal
        open={!!draftBox}
        onClose={() => setDraftBox(null)}
        dismissible={busy !== "create"}
        size="sm"
        title="New answer region"
        description={draftBox ? `Page ${pageNo} · ${draftBox[2]} × ${draftBox[3]} px at (${draftBox[0]}, ${draftBox[1]})` : undefined}
        footer={
          <>
            <Button variant="ghost" onClick={() => setDraftBox(null)} disabled={busy === "create"}>
              Cancel
            </Button>
            <Button variant="primary" iconLeft={<SquarePlus size={16} aria-hidden="true" />} loading={busy === "create"} onClick={() => void confirmDraft()}>
              Add region
            </Button>
          </>
        }
      >
        <Field label="Question" hint="The question's answer is re-read from its regions and re-graded.">
          <Select value={draftQ} onChange={(e) => setDraftQ(e.target.value)}>
            {[...questions]
              .sort((a, b) => a.question_no - b.question_no)
              .map((q) => (
                <option key={q.id} value={q.id}>
                  Q{q.question_no}
                  {q.prompt ? ` · ${q.prompt.slice(0, 60)}${q.prompt.length > 60 ? "…" : ""}` : ""}
                </option>
              ))}
            <option value="">Unassigned</option>
          </Select>
        </Field>
      </Modal>
    </div>
  );
}
