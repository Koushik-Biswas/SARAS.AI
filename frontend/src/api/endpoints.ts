// One typed function per /api/v1 route. Screens import from here, never call fetch directly.

import { downloadFile, fetchBlob, http, request, session, uploadMultipart, type UploadOptions } from "./client";
import type {
  AnalyticsOut,
  AnswerFilter,
  AnswerOut,
  AnswerSort,
  ApproveAllOut,
  AuditEntity,
  AuditOut,
  CapabilitiesOut,
  ExamIn,
  ExamOut,
  ExamPatch,
  FlagIn,
  HealthOut,
  LoginIn,
  Page,
  PageParams,
  PagesOut,
  QuestionOut,
  QuestionPaperOut,
  QuestionsIn,
  ReadyOut,
  RegionCreate,
  RegionEditOut,
  RegionOut,
  RegionPatch,
  RegisterIn,
  RegradeOut,
  ResultsOut,
  ReviewIn,
  StatusOut,
  SubmissionOut,
  SubmissionStatus,
  TokenOut,
  UploadOut,
  UserOut,
} from "./types";

const enc = encodeURIComponent;

// ---------------------------------------------------------------- auth
export async function register(body: RegisterIn): Promise<TokenOut> {
  const t = await request<TokenOut>("/auth/register", { method: "POST", body, auth: false });
  session.set(t);
  return t;
}

export async function login(body: LoginIn): Promise<TokenOut> {
  const t = await request<TokenOut>("/auth/login", { method: "POST", body, auth: false });
  session.set(t);
  return t;
}

/** Revoke the refresh token server-side (best effort) and clear the local session. */
export async function logout(): Promise<void> {
  const token = session.refreshToken;
  try {
    if (token) await request<void>("/auth/logout", { method: "POST", body: { refresh_token: token }, auth: false });
  } catch {
    /* ignore: we clear locally regardless */
  } finally {
    session.clear();
  }
}

export const me = () => http.get<UserOut>("/auth/me");

// ---------------------------------------------------------------- exams
export const listExams = (params: PageParams = {}) =>
  http.get<Page<ExamOut>>("/exams", { page: params.page, page_size: params.page_size });
export const createExam = (body: ExamIn) => http.post<ExamOut>("/exams", body);
export const getExam = (examId: string) => http.get<ExamOut>(`/exams/${enc(examId)}`);
export const updateExam = (examId: string, body: ExamPatch) => http.patch<ExamOut>(`/exams/${enc(examId)}`, body);
export const deleteExam = (examId: string) => http.del(`/exams/${enc(examId)}`);

export const getQuestions = (examId: string) => http.get<QuestionOut[]>(`/exams/${enc(examId)}/questions`);
/** Replace the whole rubric. */
export const saveQuestions = (examId: string, body: QuestionsIn) =>
  http.put<QuestionOut[]>(`/exams/${enc(examId)}/questions`, body);
/** Read an uploaded question paper (PDF / image) into a draft rubric. Nothing is saved. */
export function extractQuestions(examId: string, file: File, opts: UploadOptions = {}): Promise<QuestionPaperOut> {
  const form = new FormData();
  form.append("file", file, file.name);
  return uploadMultipart<QuestionPaperOut>(`/exams/${enc(examId)}/questions/extract`, form, opts);
}
export const regradeExam = (examId: string) => http.post<RegradeOut>(`/exams/${enc(examId)}/regrade`);

export const getResults = (examId: string) => http.get<ResultsOut>(`/exams/${enc(examId)}/results`);
export const getAnalytics = (examId: string) => http.get<AnalyticsOut>(`/exams/${enc(examId)}/analytics`);
/** Download approved results as CSV; resolves to the saved filename. */
export const exportCsv = (examId: string, fallbackName = "results.csv") =>
  downloadFile(`/exams/${enc(examId)}/export`, { format: "csv" }, fallbackName);
export const approveAll = (examId: string) => http.post<ApproveAllOut>(`/exams/${enc(examId)}/approve-all`);

// ---------------------------------------------------------------- submissions
export interface UploadSheet {
  file: File;
  /** Blank -> server uses the sanitised filename stem. */
  studentRef?: string;
}

/** Multipart upload of 1-50 PDFs. Returns 202 UploadOut (accepted + rejected files). */
export function uploadSubmissions(examId: string, sheets: UploadSheet[], opts: UploadOptions = {}): Promise<UploadOut> {
  const form = new FormData();
  for (const s of sheets) {
    form.append("files", s.file, s.file.name);
    form.append("student_refs", (s.studentRef ?? "").trim());
  }
  return uploadMultipart<UploadOut>(`/exams/${enc(examId)}/submissions`, form, opts);
}

export const listSubmissions = (
  examId: string,
  params: PageParams & { status?: SubmissionStatus | "" } = {},
  signal?: AbortSignal,
) =>
  http.get<Page<SubmissionOut>>(
    `/exams/${enc(examId)}/submissions`,
    { status: params.status, page: params.page, page_size: params.page_size },
    signal,
  );

/** Fetch every submission of an exam (walks pages of 100). */
export async function listAllSubmissions(examId: string, signal?: AbortSignal): Promise<SubmissionOut[]> {
  const out: SubmissionOut[] = [];
  for (let page = 1; page < 1000; page++) {
    const p = await listSubmissions(examId, { page, page_size: 100 }, signal);
    out.push(...p.items);
    if (out.length >= p.total || p.items.length === 0) break;
  }
  return out;
}

export const getSubmission = (id: string) => http.get<SubmissionOut>(`/submissions/${enc(id)}`);
export const getSubmissionStatus = (id: string, signal?: AbortSignal) =>
  http.get<StatusOut>(`/submissions/${enc(id)}/status`, undefined, signal);
export const retrySubmission = (id: string) => http.post<SubmissionOut>(`/submissions/${enc(id)}/retry`);
export const deleteSubmission = (id: string) => http.del(`/submissions/${enc(id)}`);
export const approveSubmission = (id: string) => http.post<SubmissionOut>(`/submissions/${enc(id)}/approve`);
export const reopenSubmission = (id: string, reason: string) =>
  http.post<SubmissionOut>(`/submissions/${enc(id)}/reopen`, { reason });

// ---------------------------------------------------------------- answers / review
export const listAnswers = (submissionId: string, opts: { status?: AnswerFilter; sort?: AnswerSort } = {}) =>
  http.get<AnswerOut[]>(`/submissions/${enc(submissionId)}/answers`, {
    status: opts.status && opts.status !== "all" ? opts.status : undefined,
    sort: opts.sort,
  });
export const getAnswer = (answerId: string) => http.get<AnswerOut>(`/answers/${enc(answerId)}`);
export const reviewAnswer = (answerId: string, body: ReviewIn) =>
  http.patch<AnswerOut>(`/answers/${enc(answerId)}/review`, body);
export const flagAnswer = (answerId: string, body: FlagIn = {}) =>
  http.post<AnswerOut>(`/answers/${enc(answerId)}/flag`, body);

// ---------------------------------------------------------------- pages / regions
export const getPages = (submissionId: string) => http.get<PagesOut>(`/submissions/${enc(submissionId)}/pages`);
/** Page image as a Blob (bearer auth). Use URL.createObjectURL and revoke when done. */
export const getPageImage = (submissionId: string, pageNo: number, thumb = false, signal?: AbortSignal) =>
  fetchBlob(`/submissions/${enc(submissionId)}/pages/${pageNo}/image`, { thumb: thumb ? 1 : 0 }, signal);

export const listRegions = (submissionId: string) =>
  http.get<RegionOut[]>(`/submissions/${enc(submissionId)}/regions`);
export const createRegion = (submissionId: string, body: RegionCreate) =>
  http.post<RegionEditOut>(`/submissions/${enc(submissionId)}/regions`, body);
export const patchRegion = (regionId: string, body: RegionPatch) =>
  http.patch<RegionEditOut>(`/regions/${enc(regionId)}`, body);
export const undoRegions = (submissionId: string) =>
  http.post<RegionEditOut>(`/submissions/${enc(submissionId)}/regions/undo`);

// ---------------------------------------------------------------- audit / system
export const listAudit = (params: { entity?: AuditEntity; id?: string } & PageParams = {}) =>
  http.get<Page<AuditOut>>("/audit-log", {
    entity: params.entity,
    id: params.id,
    page: params.page,
    page_size: params.page_size,
  });

export const getCapabilities = () => http.get<CapabilitiesOut>("/capabilities");
export const health = () => request<HealthOut>("/health", { auth: false });
export const ready = () => request<ReadyOut>("/ready", { auth: false });
