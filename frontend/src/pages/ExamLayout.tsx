import { createContext, useContext, useMemo, useState } from "react";
import { Navigate, Outlet, useLocation, useNavigate, useParams } from "react-router-dom";
import { BookOpen, Calendar, Ellipsis, Pencil, Sigma, Trash } from "lucide-react";
import { ApiError, errorMessage } from "../api/client";
import * as api from "../api/endpoints";
import type { ExamOut } from "../api/types";
import {
  Button,
  ConfirmDialog,
  DropdownMenu,
  EmptyState,
  ErrorBanner,
  IconButton,
  MenuItem,
  Skeleton,
  ButtonLink,
  useToast,
} from "../components";
import { useAsync, type AsyncState } from "../hooks/useAsync";
import { usePolling } from "../hooks/usePolling";
import { useDocumentTitle } from "../hooks/useDocumentTitle";
import { useBreadcrumbs, useExamIndex } from "../layout/ShellContext";
import { ExamFormModal } from "./ExamFormModal";
import { ANALYTICS, WORKFLOW, WorkflowStepper, nextStep } from "./workflow";

// ---------------------------------------------------------------- exam context
interface ExamContextValue {
  exam: ExamOut;
  /** Re-fetch the exam (stats drive tab badges). */
  reload: (silent?: boolean) => Promise<void>;
  setExam: (exam: ExamOut) => void;
}

const ExamContext = createContext<ExamContextValue | null>(null);

/** Whether the signed-in user may change this exam: only its owner (admins read others' exams). */
export function canEditExam(exam: ExamOut): boolean {
  return exam.can_edit !== false;
}

/** Current exam for any page rendered inside /exams/:examId/*. */
export function useExam(): ExamContextValue {
  const ctx = useContext(ExamContext);
  if (!ctx) throw new Error("useExam must be used inside the exam layout");
  return ctx;
}

/** Every exam section (the 6 workflow steps, then Analytics), for titles and breadcrumbs. */
export const EXAM_TABS = [...WORKFLOW.map(({ slug, label }) => ({ slug, label })), { slug: ANALYTICS.slug, label: ANALYTICS.label }];

export function formatDate(d: string | null | undefined): string | null {
  if (!d) return null;
  const date = new Date(d.length === 10 ? `${d}T00:00:00` : d);
  if (Number.isNaN(date.getTime())) return d;
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function HeaderSkeleton() {
  return (
    <div className="exam-header" aria-busy="true">
      <div className="stack-sm">
        <Skeleton width={320} height={30} radius={8} />
        <Skeleton width={220} height={16} />
      </div>
    </div>
  );
}

function ExamHeader({ q }: { q: AsyncState<ExamOut> & { data: ExamOut } }) {
  const exam = q.data;
  const toast = useToast();
  const navigate = useNavigate();
  const index = useExamIndex();
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const s = exam.stats;

  const onDelete = async () => {
    setDeleting(true);
    try {
      await api.deleteExam(exam.id);
      index.remove(exam.id);
      toast.success("Exam deleted", exam.title);
      navigate("/", { replace: true });
    } catch (err) {
      toast.error("Could not delete exam", errorMessage(err));
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  const date = formatDate(exam.date);
  return (
    <>
      <div className="exam-header">
        <div className="exam-header__main">
          <div className="exam-header__icon" aria-hidden="true">
            <BookOpen size={22} />
          </div>
          <div className="exam-header__text">
            <h1 className="exam-header__title">{exam.title}</h1>
            <ul className="exam-header__meta">
              {exam.subject ? <li>{exam.subject}</li> : null}
              {date ? (
                <li>
                  <Calendar size={14} aria-hidden="true" /> {date}
                </li>
              ) : null}
              <li>
                <Sigma size={14} aria-hidden="true" /> <span className="num">{exam.total_marks}</span> marks
                {s.questions ? (
                  <>
                    {" "}
                    · <span className="num">{s.questions}</span> question{s.questions === 1 ? "" : "s"}
                  </>
                ) : null}
              </li>
              <li>
                <span className="num">{s.sheets}</span> sheet{s.sheets === 1 ? "" : "s"}
              </li>
            </ul>
          </div>
        </div>
        {canEditExam(exam) ? (
          <div className="exam-header__actions">
            <Button variant="secondary" iconLeft={<Pencil size={16} aria-hidden="true" />} onClick={() => setEditing(true)}>
              Edit details
            </Button>
            <DropdownMenu
              label="More exam actions"
              trigger={(props) => <IconButton {...props} variant="secondary" label="More exam actions" icon={<Ellipsis size={18} aria-hidden="true" />} />}
            >
              {(close) => (
                <MenuItem
                  danger
                  icon={<Trash size={16} />}
                  onSelect={() => {
                    close();
                    setConfirmDelete(true);
                  }}
                >
                  Delete exam
                </MenuItem>
              )}
            </DropdownMenu>
          </div>
        ) : (
          <div className="exam-header__actions">
            <span className="muted">View only</span>
          </div>
        )}
      </div>
      <WorkflowStepper exam={exam} />

      <ExamFormModal
        open={editing}
        exam={exam}
        onClose={() => setEditing(false)}
        onSaved={(e) => {
          q.setData(e);
          index.upsert(e);
          setEditing(false);
        }}
      />
      <ConfirmDialog
        open={confirmDelete}
        tone="danger"
        title="Delete this exam?"
        message={
          <>
            <strong>{exam.title}</strong>, its rubric, all {s.sheets} uploaded sheet{s.sheets === 1 ? "" : "s"} and their marks will be permanently deleted. This cannot be undone.
          </>
        }
        confirmLabel="Delete exam"
        busy={deleting}
        onConfirm={onDelete}
        onCancel={() => setConfirmDelete(false)}
      />
    </>
  );
}

/** /exams/:examId/* - loads the exam, renders header + tab bar, provides useExam(). */
export default function ExamLayout() {
  const { examId = "" } = useParams();
  const q = useAsync(() => api.getExam(examId), [examId]);
  const index = useExamIndex();
  const { pathname } = useLocation();
  const exam = q.data;

  // Keep tab badges fresh while sheets are being processed.
  usePolling(
    async () => {
      await q.reload(true);
    },
    5000,
    !!exam && exam.stats.processing > 0,
  );

  const tab = EXAM_TABS.find((t) => pathname.startsWith(`/exams/${examId}/${t.slug}`)) ?? (pathname.includes("/regions/") ? EXAM_TABS.find((t) => t.slug === "review") : undefined);
  useDocumentTitle(exam ? `${tab ? `${tab.label} · ` : ""}${exam.title}` : "Exam");
  useBreadcrumbs(
    1,
    exam ? [{ label: exam.title, to: `/exams/${exam.id}` }, ...(tab ? [{ label: tab.label, to: `/exams/${exam.id}/${tab.slug}` }] : [])] : [{ label: "Exam" }],
  );

  const ctx = useMemo<ExamContextValue | null>(
    () =>
      exam
        ? {
            exam,
            reload: async (silent = true) => {
              await q.reload(silent);
            },
            setExam: (e: ExamOut) => {
              q.setData(e);
              index.upsert(e);
            },
          }
        : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [exam, q.reload, q.setData, index.upsert],
  );

  if (q.error && !exam) {
    const notFound = q.error instanceof ApiError && (q.error.status === 404 || q.error.status === 403);
    return (
      <div className="page">
        {notFound ? (
          <EmptyState
            illustration="exams"
            title="Exam not found"
            description="It may have been deleted, or it belongs to another teacher."
            action={<ButtonLink to="/" variant="primary">Back to dashboard</ButtonLink>}
          />
        ) : (
          <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load this exam" />
        )}
      </div>
    );
  }

  return (
    <div className="page">
      {!exam || !ctx ? (
        <>
          <HeaderSkeleton />
          <Skeleton height={64} radius={16} />
        </>
      ) : (
        <ExamContext.Provider value={ctx}>
          <ExamHeader q={q as AsyncState<ExamOut> & { data: ExamOut }} />
          <div className="exam-body">
            <Outlet />
          </div>
        </ExamContext.Provider>
      )}
    </div>
  );
}

/** /exams/:examId -> the workflow step the exam is at. */
export function ExamIndexRedirect() {
  const { exam } = useExam();
  return <Navigate to={`/exams/${exam.id}/${nextStep(exam.stats)}`} replace />;
}
