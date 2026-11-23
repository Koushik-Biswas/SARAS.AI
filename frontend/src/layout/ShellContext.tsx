import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import * as api from "../api/endpoints";
import type { ExamOut } from "../api/types";
import { useAsync } from "../hooks/useAsync";

// ---------------------------------------------------------------- exam index (sidebar + dashboard)
interface ExamIndexValue {
  exams: ExamOut[];
  total: number;
  loading: boolean;
  fetching: boolean;
  error: unknown;
  reload: (silent?: boolean) => Promise<void>;
  /** Replace/insert one exam locally (after create/update) without a refetch. */
  upsert: (exam: ExamOut) => void;
  remove: (examId: string) => void;
}

const ExamIndexContext = createContext<ExamIndexValue | null>(null);

export function ExamIndexProvider({ children }: { children: ReactNode }) {
  const q = useAsync(() => api.listExams({ page: 1, page_size: 100 }), []);
  const { setData } = q;
  const upsert = useCallback(
    (exam: ExamOut) =>
      setData((prev) => {
        const items = prev?.items ?? [];
        const exists = items.some((e) => e.id === exam.id);
        return {
          items: exists ? items.map((e) => (e.id === exam.id ? exam : e)) : [exam, ...items],
          total: (prev?.total ?? 0) + (exists ? 0 : 1),
          page: 1,
          page_size: prev?.page_size ?? 100,
        };
      }),
    [setData],
  );
  const remove = useCallback(
    (examId: string) =>
      setData((prev) => ({
        items: (prev?.items ?? []).filter((e) => e.id !== examId),
        total: Math.max(0, (prev?.total ?? 1) - 1),
        page: 1,
        page_size: prev?.page_size ?? 100,
      })),
    [setData],
  );
  const value = useMemo<ExamIndexValue>(
    () => ({
      exams: q.data?.items ?? [],
      total: q.data?.total ?? 0,
      loading: q.loading,
      fetching: q.fetching,
      error: q.error,
      reload: q.reload,
      upsert,
      remove,
    }),
    [q.data, q.loading, q.fetching, q.error, q.reload, upsert, remove],
  );
  return <ExamIndexContext.Provider value={value}>{children}</ExamIndexContext.Provider>;
}

export function useExamIndex(): ExamIndexValue {
  const ctx = useContext(ExamIndexContext);
  if (!ctx) throw new Error("useExamIndex must be used inside <ExamIndexProvider>");
  return ctx;
}

// ---------------------------------------------------------------- breadcrumbs
export interface Crumb {
  label: string;
  to?: string;
}

interface CrumbValue {
  crumbs: Crumb[];
  setCrumbs: (owner: string, crumbs: Crumb[] | null) => void;
}

const CrumbContext = createContext<CrumbValue | null>(null);

/**
 * Breadcrumbs are contributed per "level" (owner key sorted by depth): the exam layout
 * registers level 1, a nested page (e.g. Review of one sheet) may register level 2.
 */
export function CrumbProvider({ children }: { children: ReactNode }) {
  const [levels, setLevels] = useState<Record<string, Crumb[]>>({});
  const setCrumbs = useCallback((owner: string, crumbs: Crumb[] | null) => {
    setLevels((prev) => {
      const next = { ...prev };
      if (crumbs) next[owner] = crumbs;
      else delete next[owner];
      return next;
    });
  }, []);
  const crumbs = useMemo(
    () =>
      Object.keys(levels)
        .sort()
        .flatMap((k) => levels[k]),
    [levels],
  );
  const value = useMemo(() => ({ crumbs, setCrumbs }), [crumbs, setCrumbs]);
  return <CrumbContext.Provider value={value}>{children}</CrumbContext.Provider>;
}

export function useCrumbTrail(): Crumb[] {
  return useContext(CrumbContext)?.crumbs ?? [];
}

/**
 * Register breadcrumb items for this page at `level` (1 = exam, 2 = sub-page).
 * "Dashboard" is always the implicit root.
 */
export function useBreadcrumbs(level: number, crumbs: Crumb[] | null): void {
  const ctx = useContext(CrumbContext);
  const key = JSON.stringify(crumbs);
  const setCrumbs = ctx?.setCrumbs;
  useEffect(() => {
    if (!setCrumbs) return;
    const owner = `level-${String(level).padStart(2, "0")}`;
    setCrumbs(owner, crumbs);
    return () => setCrumbs(owner, null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setCrumbs, level, key]);
}
