// Side panel of the region editor: edit the selected region (reassign, crop, split, merge,
// blank), an optional reason for the next edit, the question legend and the marks that
// changed after the last edit.
import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Combine, Crop, Eraser, MousePointerClick, Scissors, SquareDashed, X } from "lucide-react";
import type { AnswerOut, BBox, PageInfo, QuestionOut, RegionOut } from "../api/types";
import { Badge, Button, ConfidenceMeter, Field, IconButton, Kbd, NumberInput, Select, StatusPill, Textarea, cx } from "../components";
import type { EditorMode } from "./RegionCanvas";
import { flagSpec, fmtMarks } from "./flags";
import { regionStyle, questionColor, type QuestionLookup } from "./regions";

export interface ChangeSummary {
  label: string;
  rows: { answer: AnswerOut; before: number | null }[];
}

interface RegionInspectorProps {
  region: RegionOut | null;
  regions: RegionOut[];
  questions: QuestionOut[];
  lookup: QuestionLookup;
  page: PageInfo | null;
  mode: EditorMode;
  readOnly: boolean;
  busy: string | null;
  reason: string;
  splitY: number | null;
  answersByQuestion: Map<string, AnswerOut>;
  lastChange: ChangeSummary | null;
  reviewHref: (answerId?: string) => string;
  onReason: (v: string) => void;
  onSplitY: (v: number | null) => void;
  onMode: (m: EditorMode) => void;
  onDeselect: () => void;
  onReassign: (questionId: string | null) => void;
  onCrop: (bbox: BBox) => void;
  onSplit: (y: number) => void;
  onMerge: (otherId: string) => void;
  onBlank: (blank: boolean) => void;
  onPickQuestion: (questionId: string | null) => void;
}

function BoxFields({ region, page, disabled, busy, onCrop }: { region: RegionOut; page: PageInfo | null; disabled: boolean; busy: boolean; onCrop: (b: BBox) => void }) {
  const [vals, setVals] = useState<(number | null)[]>(() => [...region.bbox]);
  const W = page?.width ?? 0;
  const H = page?.height ?? 0;
  const [x, y, w, h] = vals;
  const complete = x !== null && y !== null && w !== null && h !== null;
  let error: string | null = null;
  if (!complete) error = "Fill in all four values.";
  else if (w! <= 0 || h! <= 0) error = "Width and height must be positive.";
  else if (x! < 0 || y! < 0 || (W && x! + w! > W) || (H && y! + h! > H)) error = `The box must stay inside the page (${W} × ${H} px).`;
  const unchanged = complete && vals.every((v, i) => v === region.bbox[i]);
  const set = (i: number) => (v: number | null) => setVals((p) => p.map((old, j) => (j === i ? (v === null ? null : Math.round(v)) : old)));
  return (
    <div className="insp__box">
      <div className="insp__grid4">
        {(["X", "Y", "Width", "Height"] as const).map((label, i) => (
          <Field key={label} label={label}>
            <NumberInput inputSize="sm" decimals={0} step={1} min={0} value={vals[i]} onValueChange={set(i)} disabled={disabled} />
          </Field>
        ))}
      </div>
      {error && !unchanged ? <p className="field__error insp__err">{error}</p> : null}
      <Button
        size="sm"
        variant="secondary"
        block
        iconLeft={<Crop size={14} aria-hidden="true" />}
        disabled={disabled || unchanged || !!error}
        loading={busy}
        onClick={() => complete && onCrop([x!, y!, w!, h!])}
      >
        Apply box
      </Button>
    </div>
  );
}

export function RegionInspector(props: RegionInspectorProps) {
  const { region, regions, questions, lookup, page, mode, readOnly, busy, reason, splitY, answersByQuestion, lastChange, reviewHref } = props;
  const [mergeWith, setMergeWith] = useState("");
  const disabled = readOnly || !!busy;
  const sorted = [...questions].sort((a, b) => a.question_no - b.question_no);
  const counts = new Map<string, number>();
  for (const r of regions) if (r.question_id) counts.set(r.question_id, (counts.get(r.question_id) ?? 0) + 1);
  const unassigned = regions.filter((r) => !r.question_id).length;

  const legend = (
    <section className="insp__section" aria-label="Questions">
      <h3 className="insp__h">Questions</h3>
      <ul className="legend">
        {sorted.map((q, i) => {
          const a = answersByQuestion.get(q.id);
          const n = counts.get(q.id) ?? 0;
          return (
            <li key={q.id}>
              <button type="button" className="legend__item" onClick={() => props.onPickQuestion(q.id)} disabled={n === 0} title={n ? "Show this question's region" : "No region yet"}>
                <span className="legend__swatch" style={{ ["--rc" as string]: questionColor(i) }} aria-hidden="true" />
                <span className="legend__q num">Q{q.question_no}</span>
                <span className="legend__regions subtle num">
                  {n ? `${n} region${n === 1 ? "" : "s"}` : "No region"}
                </span>
                {a ? (
                  <span className="legend__marks num">
                    {fmtMarks(a.final_marks ?? a.ai_marks)}
                    <span className="subtle">/{fmtMarks(a.max_marks)}</span>
                  </span>
                ) : null}
              </button>
            </li>
          );
        })}
        {unassigned ? (
          <li>
            <button type="button" className="legend__item" onClick={() => props.onPickQuestion(null)}>
              <span className="legend__swatch legend__swatch--dashed" style={{ ["--rc" as string]: "var(--region-unassigned)" }} aria-hidden="true" />
              <span className="legend__q">Unassigned</span>
              <span className="legend__regions subtle num">
                {unassigned} region{unassigned === 1 ? "" : "s"}
              </span>
            </button>
          </li>
        ) : null}
      </ul>
    </section>
  );

  const summary = lastChange ? (
    <section className="insp__section insp__summary" aria-label="Marks after the last edit" aria-live="polite">
      <h3 className="insp__h">{lastChange.label}</h3>
      {lastChange.rows.length ? (
        <ul className="delta-list">
          {lastChange.rows.map(({ answer: a, before }) => {
            const now = a.final_marks ?? a.ai_marks;
            const changed = before !== null && Math.abs(before - now) > 1e-9;
            return (
              <li key={a.answer_id} className="delta-list__item">
                <span className="delta-list__q num">Q{a.question_no}</span>
                <span className="delta-list__marks num">
                  {changed ? (
                    <>
                      <span className="subtle">{fmtMarks(before)}</span>
                      <ArrowRight size={12} aria-label="changed to" />
                    </>
                  ) : null}
                  <strong>{fmtMarks(now)}</strong>
                  <span className="subtle">/{fmtMarks(a.max_marks)}</span>
                </span>
                <StatusPill kind="answer" status={a.status} size="sm" />
                <Link className="delta-list__link" to={reviewHref(a.answer_id)} aria-label={`Review question ${a.question_no}`}>
                  Review
                </Link>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="subtle insp__note">No answers needed re-grading.</p>
      )}
    </section>
  ) : null;

  const reasonField = !readOnly ? (
    <section className="insp__section">
      <Field label="Reason for the next edit (optional)" hint="Saved in the audit log with the change.">
        <Textarea rows={2} maxLength={500} value={reason} onChange={(e) => props.onReason(e.target.value)} />
      </Field>
    </section>
  ) : null;

  if (!region) {
    return (
      <aside className="insp" aria-label="Region details">
        <section className="insp__section insp__intro">
          <h3 className="insp__h">Edit answer regions</h3>
          <p className="muted">
            {readOnly
              ? "This sheet is read only. Select a box to see its details."
              : "Select a box to reassign, resize, split, merge or blank it, or draw a region the segmentation missed."}
          </p>
          {!readOnly ? (
            <ul className="insp__tips">
              <li>
                <Kbd>V</Kbd> select and drag · corners resize
              </li>
              <li>
                <Kbd>N</Kbd> draw a new region
              </li>
              <li>
                <Kbd>Ctrl</Kbd>
                <Kbd>Z</Kbd> undo the last edit
              </li>
            </ul>
          ) : null}
        </section>
        {summary}
        {legend}
      </aside>
    );
  }

  const st = regionStyle(region, lookup);
  const others = regions.filter((r) => r.id !== region.id);
  const [, by, , bh] = region.bbox;
  const splitValid = splitY !== null && splitY > by && splitY < by + bh;

  return (
    <aside className="insp" aria-label="Region details">
      <section className="insp__section">
        <div className="insp__head">
          <span className={cx("legend__swatch", st.dashed && "legend__swatch--dashed")} style={{ ["--rc" as string]: st.color }} aria-hidden="true" />
          <h3 className="insp__title">{st.label}</h3>
          <Badge size="sm" tone={region.source === "manual" ? "info" : "neutral"}>
            {region.source === "manual" ? "Edited" : "Auto"}
          </Badge>
          <IconButton size="sm" label="Deselect region (Esc)" icon={<X size={16} aria-hidden="true" />} onClick={props.onDeselect} className="insp__close" />
        </div>
        <div className="insp__meta">
          <span className="subtle">
            Page <span className="num">{region.page_no}</span>
          </span>
          <ConfidenceMeter value={region.segmentation_confidence} label="Segmentation confidence" compact />
        </div>
        {region.flags.length ? (
          <ul className="chip-list">
            {region.flags.map((f) => {
              const spec = flagSpec(f);
              const Icon = spec.icon;
              return (
                <li key={f}>
                  <Badge size="sm" tone={spec.tone} icon={<Icon size={11} aria-hidden="true" />}>
                    {spec.label}
                  </Badge>
                </li>
              );
            })}
          </ul>
        ) : null}
      </section>

      <section className="insp__section">
        <Field label="Question" hint="Moving a region re-grades both questions.">
          <Select
            inputSize="sm"
            value={region.question_id ?? ""}
            disabled={disabled}
            onChange={(e) => props.onReassign(e.target.value || null)}
          >
            {sorted.map((q) => (
              <option key={q.id} value={q.id}>
                Q{q.question_no}
                {q.prompt ? ` · ${q.prompt.slice(0, 48)}${q.prompt.length > 48 ? "…" : ""}` : ""}
              </option>
            ))}
            <option value="">Unassigned</option>
          </Select>
        </Field>
        <Button
          size="sm"
          variant={region.is_blank ? "secondary" : "subtle"}
          block
          iconLeft={region.is_blank ? <SquareDashed size={14} aria-hidden="true" /> : <Eraser size={14} aria-hidden="true" />}
          disabled={disabled}
          loading={busy === "blank"}
          onClick={() => props.onBlank(!region.is_blank)}
          title="Blank regions score 0 and are never sent to the grader (B)"
        >
          {region.is_blank ? "Mark as answered" : "Mark blank"} <Kbd>B</Kbd>
        </Button>
      </section>

      {!readOnly ? (
        <>
          <section className="insp__section">
            <h4 className="insp__sub">Box (page pixels)</h4>
            <p className="subtle insp__note">Drag the box to move it, or a handle to resize.</p>
            <BoxFields key={`${region.id}:${region.bbox.join(",")}`} region={region} page={page} disabled={disabled} busy={busy === "crop"} onCrop={props.onCrop} />
          </section>

          <section className="insp__section">
            <h4 className="insp__sub">Split</h4>
            <div className="insp__row">
              <Field label="Split at y" hint={`Between ${by + 1} and ${by + bh - 1}`}>
                <NumberInput
                  inputSize="sm"
                  decimals={0}
                  step={1}
                  min={by + 1}
                  max={by + bh - 1}
                  value={splitY}
                  onValueChange={props.onSplitY}
                  disabled={disabled}
                />
              </Field>
              <Button
                size="sm"
                variant="secondary"
                iconLeft={<Scissors size={14} aria-hidden="true" />}
                disabled={disabled || !splitValid}
                loading={busy === "split"}
                onClick={() => splitY !== null && props.onSplit(splitY)}
              >
                Split
              </Button>
            </div>
            <Button
              size="sm"
              variant={mode === "split" ? "primary" : "ghost"}
              block
              aria-pressed={mode === "split"}
              iconLeft={<MousePointerClick size={14} aria-hidden="true" />}
              disabled={disabled}
              onClick={() => props.onMode(mode === "split" ? "select" : "split")}
            >
              {mode === "split" ? "Click the box where to split…" : "Pick the line on the page"} <Kbd>S</Kbd>
            </Button>
          </section>

          <section className="insp__section">
            <h4 className="insp__sub">Merge</h4>
            <div className="insp__row">
              <Field label="Merge with" hideLabel>
                <Select inputSize="sm" value={mergeWith} disabled={disabled || others.length === 0} onChange={(e) => setMergeWith(e.target.value)}>
                  <option value="">Choose a region…</option>
                  {others.map((r) => {
                    const os = regionStyle(r, lookup);
                    return (
                      <option key={r.id} value={r.id}>
                        {os.label} · page {r.page_no} · y {r.bbox[1]}
                      </option>
                    );
                  })}
                </Select>
              </Field>
              <Button
                size="sm"
                variant="secondary"
                iconLeft={<Combine size={14} aria-hidden="true" />}
                disabled={disabled || !mergeWith}
                loading={busy === "merge"}
                onClick={() => {
                  if (mergeWith) {
                    props.onMerge(mergeWith);
                    setMergeWith("");
                  }
                }}
              >
                Merge
              </Button>
            </div>
            <Button
              size="sm"
              variant={mode === "merge" ? "primary" : "ghost"}
              block
              aria-pressed={mode === "merge"}
              iconLeft={<MousePointerClick size={14} aria-hidden="true" />}
              disabled={disabled || others.length === 0}
              onClick={() => props.onMode(mode === "merge" ? "select" : "merge")}
            >
              {mode === "merge" ? "Click the region to merge in…" : "Pick a region on the page"} <Kbd>M</Kbd>
            </Button>
            <p className="subtle insp__note">Same page: boxes join. Another page: that part is given this region's question.</p>
          </section>
        </>
      ) : null}

      {reasonField}
      {summary}
      {legend}
    </aside>
  );
}
