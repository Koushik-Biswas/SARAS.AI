// Interactive page canvas for the region editor. All geometry is in page-image pixels
// (the image's naturalWidth/Height); pointer positions are converted from the displayed size.
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { PencilLine } from "lucide-react";
import type { BBox, PageInfo, RegionOut } from "../api/types";
import { cx } from "../components";
import { Paper, type PageSpace } from "./Paper";
import {
  HANDLES,
  MIN_BOX,
  boxStyle,
  clamp,
  clampBox,
  describeRegion,
  dragBox,
  paintOrder,
  rectFromPoints,
  regionStyle,
  sameBox,
  type Handle,
  type QuestionLookup,
} from "./regions";

export type EditorMode = "select" | "draw" | "split" | "merge";

interface RegionCanvasProps {
  submissionId: string;
  page: PageInfo;
  /** Regions on this page. */
  regions: RegionOut[];
  lookup: QuestionLookup;
  selectedId: string | null;
  mode: EditorMode;
  readOnly: boolean;
  busy: boolean;
  zoom: number;
  /** Rectangle drawn and awaiting confirmation. */
  draftBox: BBox | null;
  /** Split position typed in the inspector (page px), previewed as a guide line. */
  splitY: number | null;
  onSelect: (id: string | null) => void;
  onCancelMode: () => void;
  onCommitBox: (id: string, bbox: BBox) => Promise<unknown>;
  onSplit: (id: string, y: number) => void;
  onDrawn: (bbox: BBox) => void;
  onMergePick: (id: string) => void;
  onSpace?: (space: PageSpace) => void;
}

interface Drag {
  id: string;
  handle: Handle | null;
  orig: BBox;
  /** Start point in page px and in client px (for the click-vs-drag threshold). */
  px: number;
  py: number;
  cx: number;
  cy: number;
  moved: boolean;
  pointerId: number;
}

export function RegionCanvas({
  submissionId,
  page,
  regions,
  lookup,
  selectedId,
  mode,
  readOnly,
  busy,
  zoom,
  draftBox,
  splitY,
  onSelect,
  onCancelMode,
  onCommitBox,
  onSplit,
  onDrawn,
  onMergePick,
  onSpace,
}: RegionCanvasProps) {
  const paperRef = useRef<HTMLDivElement>(null);
  const spaceRef = useRef<PageSpace>({ w: page.width, h: page.height });
  const dragRef = useRef<Drag | null>(null);
  const drawRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const [preview, setPreview] = useState<{ id: string; bbox: BBox } | null>(null);
  const [drawRect, setDrawRect] = useState<BBox | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);
  const editable = !readOnly && !busy;
  const selected = regions.find((r) => r.id === selectedId) ?? null;

  useEffect(() => {
    spaceRef.current = { w: page.width, h: page.height };
  }, [page.width, page.height]);
  useEffect(() => setHoverY(null), [mode, selectedId]);

  // Esc cancels an in-progress drag or rectangle without committing anything.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (dragRef.current || drawRef.current) {
        dragRef.current = null;
        drawRef.current = null;
        setPreview(null);
        setDrawRect(null);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  const toPage = (clientX: number, clientY: number) => {
    const el = paperRef.current;
    const s = spaceRef.current;
    if (!el || !s.w || !s.h) return { x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    return {
      x: clamp(((clientX - r.left) * s.w) / r.width, 0, s.w),
      y: clamp(((clientY - r.top) * s.h) / r.height, 0, s.h),
    };
  };

  const capture = (pointerId: number) => {
    try {
      paperRef.current?.setPointerCapture(pointerId);
    } catch {
      /* pointer already released */
    }
  };
  const release = (pointerId: number) => {
    const el = paperRef.current;
    if (el?.hasPointerCapture(pointerId)) el.releasePointerCapture(pointerId);
  };

  const boxOf = (r: RegionOut): BBox => (preview && preview.id === r.id ? preview.bbox : r.bbox);

  // ---------------------------------------------------------------- pointer handlers
  const onPaperDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if (mode === "draw") {
      if (!editable) return;
      e.preventDefault();
      const p = toPage(e.clientX, e.clientY);
      drawRef.current = { x: p.x, y: p.y, pointerId: e.pointerId };
      setDrawRect([p.x, p.y, 0, 0]);
      capture(e.pointerId);
      return;
    }
    if (mode === "select") onSelect(null);
    else onCancelMode();
  };

  const startDrag = (e: PointerEvent<HTMLElement>, r: RegionOut, handle: Handle | null) => {
    e.preventDefault();
    const p = toPage(e.clientX, e.clientY);
    dragRef.current = { id: r.id, handle, orig: r.bbox, px: p.x, py: p.y, cx: e.clientX, cy: e.clientY, moved: false, pointerId: e.pointerId };
    capture(e.pointerId);
  };

  const onBoxDown = (e: PointerEvent<HTMLDivElement>, r: RegionOut) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    if (mode === "merge") {
      if (selectedId && r.id !== selectedId && editable) onMergePick(r.id);
      return;
    }
    if (mode === "split") {
      if (r.id === selectedId && editable) {
        const y = Math.round(toPage(e.clientX, e.clientY).y);
        if (y > r.bbox[1] && y < r.bbox[1] + r.bbox[3]) onSplit(r.id, y);
      } else onSelect(r.id);
      return;
    }
    onSelect(r.id);
    if (editable && mode === "select") startDrag(e, r, null);
  };

  const onHandleDown = (e: PointerEvent<HTMLSpanElement>, r: RegionOut, h: Handle) => {
    if (e.button !== 0 || !editable) return;
    e.stopPropagation();
    startDrag(e, r, h);
  };

  const onPaperMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = dragRef.current;
    const s = spaceRef.current;
    if (d) {
      if (!d.moved && Math.hypot(e.clientX - d.cx, e.clientY - d.cy) < 3) return;
      d.moved = true;
      const p = toPage(e.clientX, e.clientY);
      setPreview({ id: d.id, bbox: dragBox(d.orig, d.handle, p.x - d.px, p.y - d.py, s.w, s.h) });
      return;
    }
    const dr = drawRef.current;
    if (dr) {
      const p = toPage(e.clientX, e.clientY);
      setDrawRect(rectFromPoints(dr.x, dr.y, p.x, p.y));
      return;
    }
    if (mode === "split" && selected && editable) {
      const p = toPage(e.clientX, e.clientY);
      const [x, y, w, h] = selected.bbox;
      const inside = p.x >= x && p.x <= x + w && p.y > y + 1 && p.y < y + h - 1;
      setHoverY(inside ? Math.round(p.y) : null);
    }
  };

  const onPaperUp = (e: PointerEvent<HTMLDivElement>) => {
    const s = spaceRef.current;
    const d = dragRef.current;
    if (d) {
      dragRef.current = null;
      release(d.pointerId);
      if (!d.moved) {
        setPreview(null);
        return;
      }
      const p = toPage(e.clientX, e.clientY);
      const box = dragBox(d.orig, d.handle, p.x - d.px, p.y - d.py, s.w, s.h);
      if (sameBox(box, d.orig)) {
        setPreview(null);
        return;
      }
      setPreview({ id: d.id, bbox: box });
      void Promise.resolve(onCommitBox(d.id, box)).finally(() => setPreview((pv) => (pv && pv.id === d.id ? null : pv)));
      return;
    }
    const dr = drawRef.current;
    if (dr) {
      drawRef.current = null;
      release(dr.pointerId);
      const p = toPage(e.clientX, e.clientY);
      const raw = rectFromPoints(dr.x, dr.y, p.x, p.y);
      setDrawRect(null);
      if (raw[2] >= MIN_BOX * 2 && raw[3] >= MIN_BOX * 2) onDrawn(clampBox(raw, s.w, s.h));
    }
  };

  const onPaperCancel = () => {
    dragRef.current = null;
    drawRef.current = null;
    setPreview(null);
    setDrawRect(null);
  };

  const onBoxKey = (e: KeyboardEvent<HTMLDivElement>, r: RegionOut) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    if (mode === "merge" && selectedId && r.id !== selectedId) {
      if (editable) onMergePick(r.id);
    } else onSelect(r.id);
  };

  // ---------------------------------------------------------------- render
  const guideY = mode === "split" && selected ? hoverY ?? splitY : null;
  const showDraft = drawRect ?? draftBox;

  return (
    <Paper
      submissionId={submissionId}
      page={page}
      zoom={zoom}
      lazy={false}
      alt={`Scanned page ${page.page_no}`}
      className={cx("rcanvas__paper", `rcanvas--${mode}`, readOnly && "rcanvas--readonly")}
      paperRef={paperRef}
      onSpace={(s) => {
        spaceRef.current = s;
        onSpace?.(s);
      }}
      paperProps={{
        onPointerDown: onPaperDown,
        onPointerMove: onPaperMove,
        onPointerUp: onPaperUp,
        onPointerCancel: onPaperCancel,
        onPointerLeave: () => setHoverY(null),
      }}
    >
      {(space) => (
        <>
          {paintOrder(regions).map((r) => {
            const st = regionStyle(r, lookup);
            const box = boxOf(r);
            const isSel = r.id === selectedId;
            const target = mode === "merge" && !!selectedId && !isSel;
            const nearTop = box[1] / space.h < 0.03;
            return (
              <div
                key={r.id}
                data-region-id={r.id}
                role="button"
                tabIndex={0}
                aria-pressed={isSel}
                aria-label={describeRegion({ ...r, bbox: box }, st.label) + (target ? ". Press Enter to merge into the selected region" : "")}
                className={cx(
                  "rbox",
                  st.dashed && "rbox--dashed",
                  isSel && "is-selected",
                  target && "is-target",
                  preview?.id === r.id && "is-dragging",
                )}
                style={{ ...boxStyle(box, space.w, space.h), ["--rc" as string]: st.color }}
                onPointerDown={(e) => onBoxDown(e, r)}
                onKeyDown={(e) => onBoxKey(e, r)}
              >
                <span className={cx("rbox__label", nearTop && "rbox__label--inside")}>
                  {st.label}
                  {r.source === "manual" ? <PencilLine size={10} aria-hidden="true" /> : null}
                </span>
                {isSel && mode === "select" && editable
                  ? HANDLES.map((h) => (
                      <span
                        key={h}
                        className={`rbox__handle rbox__handle--${h}`}
                        aria-hidden="true"
                        onPointerDown={(e) => onHandleDown(e, r, h)}
                      />
                    ))
                  : null}
              </div>
            );
          })}
          {guideY !== null && selected ? (
            <div
              className="rsplit"
              aria-hidden="true"
              style={{
                top: `${(guideY / space.h) * 100}%`,
                left: `${(selected.bbox[0] / space.w) * 100}%`,
                width: `${(selected.bbox[2] / space.w) * 100}%`,
              }}
            >
              <span className="rsplit__label num">Split at y = {guideY}</span>
            </div>
          ) : null}
          {showDraft && showDraft[2] > 0 && showDraft[3] > 0 ? (
            <div className="rdraft" aria-hidden="true" style={boxStyle(showDraft, space.w, space.h)}>
              <span className="rdraft__label num">
                {Math.round(showDraft[2])} × {Math.round(showDraft[3])}
              </span>
            </div>
          ) : null}
        </>
      )}
    </Paper>
  );
}
