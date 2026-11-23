import { useMemo, useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, CircleX, CloudUpload, FileText, ListChecks, Trash, Upload, X } from "lucide-react";
import { ApiError, errorMessage, newIdempotencyKey } from "../api/client";
import * as api from "../api/endpoints";
import type { UploadOut, UploadRejection } from "../api/types";
import { Banner, Button, ButtonLink, Card, EmptyState, Field, IconButton, Input, Kbd, ProgressBar, cx, useToast } from "../components";
import { useAsync } from "../hooks/useAsync";
import { canEditExam, useExam } from "./ExamLayout";
import { StepFooter } from "./workflow";

const MAX_FILES = 50;
const DEFAULT_MAX_MB = 20;

interface PendingFile {
  id: string;
  file: File;
  studentRef: string;
}

interface LocalRejection {
  name: string;
  reason: string;
}

export function studentRefFromFilename(name: string): string {
  const stem = name.replace(/\.pdf$/i, "");
  return stem
    .replace(/[^A-Za-z0-9 _.-]+/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 64);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const CODE_TEXT: Record<string, string> = {
  unsupported_media_type: "Not a PDF",
  file_too_large: "Too large",
  unreadable_pdf: "Unreadable PDF",
  conflict: "Duplicate",
  validation_error: "Invalid",
};

function RejectionList({ items }: { items: UploadRejection[] }) {
  return (
    <ul className="reject-list">
      {items.map((r, i) => (
        <li key={`${r.filename}-${i}`}>
          <CircleX size={16} aria-hidden="true" className="reject-list__icon" />
          <div className="grow">
            <p className="reject-list__name">
              <span className="truncate">{r.filename}</span>
              {r.student_ref ? <span className="subtle"> · {r.student_ref}</span> : null}
            </p>
            <p className="reject-list__msg">
              <strong>{CODE_TEXT[r.code] ?? r.code.replace(/_/g, " ")}:</strong> {r.message}
            </p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function UploadPage() {
  const { exam, reload: reloadExam } = useExam();
  const toast = useToast();
  const caps = useAsync(() => api.getCapabilities().catch(() => null), []);
  const maxMb = caps.data?.settings.max_upload_mb ?? DEFAULT_MAX_MB;
  const maxPages = caps.data?.settings.max_pages ?? 50;

  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<PendingFile[]>([]);
  const [localRejects, setLocalRejects] = useState<LocalRejection[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [result, setResult] = useState<UploadOut | null>(null);
  // Files the server's answer did not mention: a retried request (same Idempotency-Key) only
  // returns the sheets it created, so files rejected by a first attempt whose response was
  // lost are not listed anywhere. They stay in the list to upload again.
  const [unreported, setUnreported] = useState<string[]>([]);
  const [failure, setFailure] = useState<{ message: string; rejected: UploadRejection[] } | null>(null);
  // Same key for retries of the same batch so the server never creates duplicates.
  const keyRef = useRef<string | null>(null);

  const addFiles = (list: FileList | File[]) => {
    const incoming = Array.from(list);
    const rejects: LocalRejection[] = [];
    const accepted: PendingFile[] = [];
    const existing = new Set(files.map((f) => `${f.file.name}:${f.file.size}`));
    for (const file of incoming) {
      const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
      if (!isPdf) rejects.push({ name: file.name, reason: "is not a PDF. Only PDF scans can be uploaded." });
      else if (file.size > maxMb * 1024 * 1024) rejects.push({ name: file.name, reason: `is ${formatBytes(file.size)}; the limit is ${maxMb} MB per file.` });
      else if (file.size === 0) rejects.push({ name: file.name, reason: "is empty." });
      else if (existing.has(`${file.name}:${file.size}`)) rejects.push({ name: file.name, reason: "is already in the list." });
      else if (files.length + accepted.length >= MAX_FILES) rejects.push({ name: file.name, reason: `was not added: at most ${MAX_FILES} files per upload.` });
      else {
        existing.add(`${file.name}:${file.size}`);
        accepted.push({ id: newIdempotencyKey(), file, studentRef: studentRefFromFilename(file.name) });
      }
    }
    setLocalRejects(rejects);
    if (accepted.length) {
      setFiles((xs) => [...xs, ...accepted]);
      setResult(null);
      setFailure(null);
      keyRef.current = null;
    }
  };

  const dupRefs = useMemo(() => {
    const counts = new Map<string, number>();
    files.forEach((f) => {
      const k = f.studentRef.trim().toLowerCase();
      if (k) counts.set(k, (counts.get(k) ?? 0) + 1);
    });
    return new Set([...counts].filter(([, n]) => n > 1).map(([k]) => k));
  }, [files]);
  const totalBytes = files.reduce((a, f) => a + f.file.size, 0);
  const blocked = files.length === 0 || dupRefs.size > 0 || uploading;

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const upload = async () => {
    if (blocked) return;
    keyRef.current ??= newIdempotencyKey();
    setUploading(true);
    setProgress(0);
    setFailure(null);
    setResult(null);
    setUnreported([]);
    const sent = files;
    try {
      const out = await api.uploadSubmissions(
        exam.id,
        sent.map((f) => ({ file: f.file, studentRef: f.studentRef })),
        { idempotencyKey: keyRef.current, onProgress: (p) => setProgress(p.fraction) },
      );
      // Every file sent must come back as a sheet or a rejection; match by file name.
      const reported = new Map<string, number>();
      for (const name of [...out.submissions.map((s) => s.original_filename ?? ""), ...out.rejected.map((r) => r.filename)]) {
        reported.set(name, (reported.get(name) ?? 0) + 1);
      }
      const missing = sent.filter((f) => {
        const n = reported.get(f.file.name) ?? 0;
        if (n > 0) reported.set(f.file.name, n - 1);
        return n === 0;
      });
      const unaccounted = out.submissions.length + out.rejected.length < sent.length ? missing : [];
      setResult(out);
      setUnreported(unaccounted.map((f) => f.file.name));
      setFiles(unaccounted); // keep them in the list so they can be sent again
      keyRef.current = null;
      const n = out.submissions.length;
      if (unaccounted.length) toast.error(`${n} sheet${n === 1 ? "" : "s"} uploaded`, `${unaccounted.length} file${unaccounted.length === 1 ? " was" : "s were"} not processed. They are still in the list: upload them again.`);
      else if (out.rejected.length) toast.info(`${n} sheet${n === 1 ? "" : "s"} uploaded`, `${out.rejected.length} file${out.rejected.length === 1 ? " was" : "s were"} rejected. See the list for details.`);
      else toast.success(`${n} sheet${n === 1 ? "" : "s"} uploaded`, "Processing has started. Follow progress on the Status tab.");
      void reloadExam(true);
    } catch (err) {
      const rejected = err instanceof ApiError && Array.isArray(err.details) ? (err.details as UploadRejection[]).filter((d) => d && typeof d === "object" && "filename" in d) : [];
      setFailure({ message: errorMessage(err), rejected });
      if (err instanceof ApiError && err.status !== 0) keyRef.current = null; // server answered: a new attempt is a new batch
      toast.error("Upload failed", errorMessage(err));
    } finally {
      setUploading(false);
      setProgress(null);
    }
  };

  if (!canEditExam(exam)) {
    return (
      <Card>
        <EmptyState
          illustration="upload"
          title="View only"
          description="Only the teacher who owns this exam can upload answer sheets to it."
          action={
            <ButtonLink to={`/exams/${exam.id}/status`} variant="secondary">
              See uploaded sheets
            </ButtonLink>
          }
        />
      </Card>
    );
  }

  if (exam.stats.questions === 0) {
    return (
      <Card>
        <EmptyState
          illustration="upload"
          title="Add the rubric first"
          description="Sheets are graded against the rubric, so create at least one question before uploading."
          action={
            <ButtonLink to={`/exams/${exam.id}/rubric`} variant="primary" iconLeft={<ListChecks size={16} aria-hidden="true" />}>
              Go to rubric
            </ButtonLink>
          }
        />
      </Card>
    );
  }

  return (
    <div className="stack">
    <div className="upload-layout">
      <div className="stack">
        <div
          className={cx("dropzone", dragging && "is-dragging", uploading && "is-disabled")}
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragging(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
          }}
          onDrop={(e) => (uploading ? e.preventDefault() : onDrop(e))}
        >
          <div className="dropzone__icon" aria-hidden="true">
            <CloudUpload size={28} />
          </div>
          <p className="dropzone__title">Drag and drop scanned answer sheets</p>
          <p className="dropzone__text">
            One PDF per student · up to {maxMb} MB and {maxPages} pages each · {MAX_FILES} files per upload
          </p>
          <Button variant="primary" iconLeft={<Upload size={16} aria-hidden="true" />} onClick={() => inputRef.current?.click()} disabled={uploading}>
            Choose PDF files
          </Button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            className="sr-only"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {localRejects.length ? (
          <Banner
            tone="warning"
            title={`${localRejects.length} file${localRejects.length === 1 ? " was" : "s were"} not added`}
            action={<IconButton size="sm" label="Dismiss" icon={<X size={16} aria-hidden="true" />} onClick={() => setLocalRejects([])} />}
          >
            <ul className="plain-list">
              {localRejects.map((r, i) => (
                <li key={i}>
                  <strong>{r.name}</strong> {r.reason}
                </li>
              ))}
            </ul>
          </Banner>
        ) : null}

        {failure ? (
          <Banner tone="danger" title="Upload failed">
            <p>{failure.message}</p>
            {failure.rejected.length ? <RejectionList items={failure.rejected} /> : null}
          </Banner>
        ) : null}

        {result ? (
          <Card
            title={`${result.submissions.length} sheet${result.submissions.length === 1 ? "" : "s"} uploaded`}
            subtitle="Sheets are processed in the background. You can keep working."
            actions={
              <ButtonLink to={`/exams/${exam.id}/status`} variant="primary" size="sm" iconRight={<ArrowRight size={14} aria-hidden="true" />}>
                View status
              </ButtonLink>
            }
          >
            {unreported.length ? (
              <Banner tone="warning" title={`${unreported.length} file${unreported.length === 1 ? " was" : "s were"} not processed`}>
                <p>
                  The server did not report on {unreported.length === 1 ? "this file" : "these files"} (an earlier attempt's reply was probably lost). {unreported.length === 1 ? "It is" : "They are"} still in the list below: upload again to get a result.
                </p>
                <ul className="plain-list">
                  {unreported.map((name, i) => (
                    <li key={`${name}-${i}`}>
                      <strong>{name}</strong>
                    </li>
                  ))}
                </ul>
              </Banner>
            ) : null}
            {result.rejected.length ? (
              <div className="stack-sm">
                <p className="section-label">Rejected by the server</p>
                <RejectionList items={result.rejected} />
              </div>
            ) : unreported.length ? null : (
              <p className="muted">All files were accepted.</p>
            )}
          </Card>
        ) : null}

        {files.length > 0 ? (
          <Card
            title={`Ready to upload (${files.length})`}
            subtitle={`${formatBytes(totalBytes)} in total. Check each student ID; it defaults to the file name.`}
            actions={
              <Button variant="ghost" size="sm" onClick={() => setFiles([])} disabled={uploading}>
                Clear all
              </Button>
            }
            flush
          >
            <ul className="file-list">
              {files.map((f) => {
                const dup = dupRefs.has(f.studentRef.trim().toLowerCase());
                return (
                  <li key={f.id} className="file-row">
                    <span className="file-row__icon" aria-hidden="true">
                      <FileText size={18} />
                    </span>
                    <div className="file-row__info">
                      <p className="file-row__name truncate" title={f.file.name}>
                        {f.file.name}
                      </p>
                      <p className="file-row__size num">{formatBytes(f.file.size)}</p>
                    </div>
                    <Field label={`Student ID for ${f.file.name}`} hideLabel error={dup ? "Duplicate student ID" : undefined} className="file-row__ref">
                      <Input
                        inputSize="sm"
                        value={f.studentRef}
                        maxLength={64}
                        disabled={uploading}
                        placeholder="Student ID"
                        onChange={(e) => {
                          const v = e.target.value;
                          setFiles((xs) => xs.map((x) => (x.id === f.id ? { ...x, studentRef: v } : x)));
                          keyRef.current = null;
                        }}
                      />
                    </Field>
                    <IconButton
                      size="sm"
                      label={`Remove ${f.file.name}`}
                      icon={<Trash size={15} aria-hidden="true" />}
                      disabled={uploading}
                      onClick={() => {
                        setFiles((xs) => xs.filter((x) => x.id !== f.id));
                        keyRef.current = null;
                      }}
                    />
                  </li>
                );
              })}
            </ul>
            <div className="upload-actions">
              {uploading ? (
                <ProgressBar
                  className="grow"
                  label="Uploading sheets"
                  showLabel
                  value={progress !== null && progress < 1 ? progress : null}
                  valueText={progress !== null && progress < 1 ? `${Math.round(progress * 100)}%` : "Validating on server..."}
                />
              ) : (
                <p className="muted grow">{dupRefs.size ? "Give every sheet a unique student ID." : "Blank IDs use the file name."}</p>
              )}
              <Button variant="primary" iconLeft={<Upload size={16} aria-hidden="true" />} onClick={upload} disabled={blocked} loading={uploading}>
                Upload {files.length} sheet{files.length === 1 ? "" : "s"}
              </Button>
            </div>
          </Card>
        ) : null}
      </div>

      <aside className="stack">
        <Card title="Scanning tips">
          <ul className="tips">
            <li>Scan at 200-300 DPI, in greyscale or colour.</li>
            <li>Students should start each answer with its marker, e.g. <Kbd>Q1</Kbd> or <Kbd>1.</Kbd></li>
            <li>Password-protected PDFs are rejected.</li>
            <li>A bad page never fails the rest of the sheet; it is flagged for review instead.</li>
          </ul>
        </Card>
        <Card title="Exam so far">
          <dl className="mini-stats">
            <div>
              <dt>Sheets uploaded</dt>
              <dd className="num">{exam.stats.sheets}</dd>
            </div>
            <div>
              <dt>Processing</dt>
              <dd className="num">{exam.stats.processing}</dd>
            </div>
            <div>
              <dt>Failed</dt>
              <dd className="num">{exam.stats.failed}</dd>
            </div>
          </dl>
          {exam.stats.sheets ? (
            <Link to={`/exams/${exam.id}/status`} className="inline-link">
              Open status <ArrowRight size={14} aria-hidden="true" />
            </Link>
          ) : null}
        </Card>
      </aside>
    </div>
      <StepFooter exam={exam} slug="upload" hint={exam.stats.sheets ? "The AI engines start checking as soon as sheets arrive" : undefined} />
    </div>
  );
}
