// Region helpers: per-question colours (tokens), labels and bbox maths in page-image pixels.
import type { BBox, RegionOut } from "../api/types";

export const PALETTE_SIZE = 8;
/** Smallest box (page px) the editor will create or resize to. */
export const MIN_BOX = 8;

/** CSS colour (a token reference) for the question at `index` in rubric order; null = unassigned. */
export function questionColor(index: number | null | undefined): string {
  if (index === null || index === undefined || index < 0) return "var(--region-unassigned)";
  return `var(--region-q${(index % PALETTE_SIZE) + 1})`;
}

/** Anything with a question id + number (rubric questions, or answers mapped to {id: question_id}). */
export interface QuestionRef {
  id: string;
  question_no: number;
}

export interface QuestionLookup {
  noOf: Map<string, number>;
  indexOf: Map<string, number>;
}

export function questionLookup(questions: QuestionRef[]): QuestionLookup {
  const sorted = [...questions].sort((a, b) => a.question_no - b.question_no);
  return {
    noOf: new Map(sorted.map((q) => [q.id, q.question_no])),
    indexOf: new Map(sorted.map((q, i) => [q.id, i])),
  };
}

export interface RegionStyle {
  color: string;
  label: string;
  dashed: boolean;
}

/** Label is always text ("Q3", "Q3 · Blank", "Unassigned"), so colour is never the only cue. */
export function regionStyle(r: RegionOut, lookup: QuestionLookup | null): RegionStyle {
  const no = r.question_id && lookup ? lookup.noOf.get(r.question_id) ?? null : null;
  const idx = r.question_id && lookup ? lookup.indexOf.get(r.question_id) ?? null : null;
  if (!r.question_id || no === null) {
    return { color: "var(--region-unassigned)", label: r.is_blank ? "Blank" : "Unassigned", dashed: true };
  }
  return {
    color: r.is_blank ? "var(--region-blank)" : questionColor(idx),
    label: r.is_blank ? `Q${no} · Blank` : `Q${no}`,
    dashed: r.is_blank,
  };
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Integer box fully inside a W x H page with at least MIN_BOX size. */
export function clampBox([x, y, w, h]: BBox, W: number, H: number): BBox {
  const x0 = clamp(Math.round(x), 0, Math.max(0, W - MIN_BOX));
  const y0 = clamp(Math.round(y), 0, Math.max(0, H - MIN_BOX));
  const x1 = clamp(Math.round(x + w), x0 + MIN_BOX, W);
  const y1 = clamp(Math.round(y + h), y0 + MIN_BOX, H);
  return [x0, y0, x1 - x0, y1 - y0];
}

export function rectFromPoints(ax: number, ay: number, bx: number, by: number): BBox {
  const x = Math.min(ax, bx);
  const y = Math.min(ay, by);
  return [x, y, Math.abs(bx - ax), Math.abs(by - ay)];
}

export function sameBox(a: BBox, b: BBox): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

export type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
export const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** Apply a move (handle = null) or a resize drag of (dx, dy) page px to `orig`, kept inside the page. */
export function dragBox(orig: BBox, handle: Handle | null, dx: number, dy: number, W: number, H: number): BBox {
  const [x, y, w, h] = orig;
  if (handle === null) {
    const nx = clamp(Math.round(x + dx), 0, Math.max(0, W - w));
    const ny = clamp(Math.round(y + dy), 0, Math.max(0, H - h));
    return [nx, ny, Math.min(w, W), Math.min(h, H)];
  }
  let x0 = x;
  let y0 = y;
  let x1 = x + w;
  let y1 = y + h;
  if (handle.includes("w")) x0 = clamp(x + dx, 0, x1 - MIN_BOX);
  if (handle.includes("e")) x1 = clamp(x1 + dx, x0 + MIN_BOX, W);
  if (handle.includes("n")) y0 = clamp(y + dy, 0, y1 - MIN_BOX);
  if (handle.includes("s")) y1 = clamp(y1 + dy, y0 + MIN_BOX, H);
  const rx = Math.round(x0);
  const ry = Math.round(y0);
  return [rx, ry, Math.round(x1) - rx, Math.round(y1) - ry];
}

/** Percentage CSS box for absolutely-positioned overlays (scales with the displayed image). */
export function boxStyle([x, y, w, h]: BBox, W: number, H: number): { left: string; top: string; width: string; height: string } {
  const p = (v: number, t: number) => `${t > 0 ? (v / t) * 100 : 0}%`;
  return { left: p(x, W), top: p(y, H), width: p(w, W), height: p(h, H) };
}

/** Regions sorted for painting: big boxes first so small ones stay clickable on top. */
export function paintOrder(regions: RegionOut[]): RegionOut[] {
  return [...regions].sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3]);
}

export function describeRegion(r: RegionOut, label: string): string {
  const [x, y, w, h] = r.bbox;
  return `${label}, page ${r.page_no}, ${r.source === "manual" ? "edited by teacher" : "detected automatically"}, at ${x}, ${y}, size ${w} by ${h} pixels`;
}
