// Left pane of the review workspace: the scanned page(s) holding the current answer, with
// its regions outlined. Other answers' regions on the same page are drawn muted for context.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Crop, FileText } from "lucide-react";
import type { AnswerOut, PageInfo, RegionOut } from "../api/types";
import { Badge, ButtonLink, EmptyState, ErrorBanner, Segmented, Skeleton, cx } from "../components";
import { ZoomControls } from "./controls";
import { flagSpec } from "./flags";
import { prefersReducedMotion } from "./hooks";
import { Paper } from "./Paper";
import { boxStyle, paintOrder, regionStyle, type QuestionLookup } from "./regions";

type Scope = "answer" | "all";

interface ScanViewerProps {
  submissionId: string;
  pages: PageInfo[] | undefined;
  pagesError: unknown;
  onRetry: () => void;
  regions: RegionOut[] | undefined;
  answer: AnswerOut | null;
  lookup: QuestionLookup;
  zoom: number;
  onZoom: (z: number) => void;
  regionsHref: string;
  canEditRegions: boolean;
}

export function ScanViewer({
  submissionId,
  pages,
  pagesError,
  onRetry,
  regions,
  answer,
  lookup,
  zoom,
  onZoom,
  regionsHref,
  canEditRegions,
}: ScanViewerProps) {
  const [scope, setScope] = useState<Scope>("answer");
  const scrollRef = useRef<HTMLDivElement>(null);

  const answerRegionIds = useMemo(() => new Set(answer?.region_ids ?? []), [answer]);
  const answerPages = useMemo(() => {
    const set = new Set((regions ?? []).filter((r) => answerRegionIds.has(r.id)).map((r) => r.page_no));
    return [...set].sort((a, b) => a - b);
  }, [regions, answerRegionIds]);
  const noRegion = !!answer && answerPages.length === 0;
  const effectiveScope: Scope = noRegion ? "all" : scope;
  const shown = useMemo(
    () => (pages ?? []).filter((p) => effectiveScope === "all" || answerPages.includes(p.page_no)),
    [pages, effectiveScope, answerPages],
  );
  const regionsByPage = useMemo(() => {
    const m = new Map<number, RegionOut[]>();
    for (const r of regions ?? []) {
      const list = m.get(r.page_no) ?? [];
      list.push(r);
      m.set(r.page_no, list);
    }
    return m;
  }, [regions]);

  // Bring the answer's first region into view whenever the answer (or scope / zoom) changes.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const raf = requestAnimationFrame(() => {
      const el = container.querySelector<HTMLElement>('[data-focus-box="true"]');
      const behavior: ScrollBehavior = prefersReducedMotion() ? "auto" : "smooth";
      if (!el) {
        container.scrollTo({ top: 0, behavior });
        return;
      }
      const cr = container.getBoundingClientRect();
      const er = el.getBoundingClientRect();
      const tall = er.height > cr.height * 0.7;
      const top = container.scrollTop + (er.top - cr.top) - (tall ? 32 : cr.height / 2 - er.height / 2);
      const left = container.scrollLeft + (er.left - cr.left) - Math.max(16, cr.width / 2 - er.width / 2);
      container.scrollTo({ top: Math.max(0, top), left: zoom > 1 ? Math.max(0, left) : 0, behavior });
    });
    return () => cancelAnimationFrame(raf);
  }, [answer?.answer_id, effectiveScope, zoom, shown.length, regions]);

  const pageSummary =
    !answer || !pages
      ? ""
      : noRegion
        ? `All ${pages.length} page${pages.length === 1 ? "" : "s"}`
        : effectiveScope === "all"
          ? `All ${pages.length} pages · answer on ${answerPages.length === 1 ? `page ${answerPages[0]}` : `pages ${answerPages.join(", ")}`}`
          : answerPages.length === 1
            ? `Page ${answerPages[0]} of ${pages.length}`
            : `Pages ${answerPages.join(", ")} of ${pages.length}`;

  return (
    <section className="pane pane--scan" aria-label="Scanned answer">
      <div className="pane__toolbar">
        <div className="pane__title">
          <FileText size={16} aria-hidden="true" />
          <span className="truncate">{pageSummary || "Scan"}</span>
        </div>
        <div className="pane__tools">
          {!noRegion && pages && pages.length > 1 ? (
            <Segmented<Scope>
              size="sm"
              label="Pages shown"
              value={scope}
              onChange={setScope}
              options={[
                { value: "answer", label: "Answer" },
                { value: "all", label: "All pages" },
              ]}
            />
          ) : null}
          <ZoomControls zoom={zoom} onZoom={onZoom} />
          {canEditRegions ? (
            <ButtonLink to={regionsHref} size="sm" variant="ghost" iconLeft={<Crop size={14} aria-hidden="true" />}>
              Edit regions
            </ButtonLink>
          ) : null}
        </div>
      </div>
      <div className="pane__body viewer" ref={scrollRef} tabIndex={0} data-own-arrows aria-label="Scanned pages (scrollable)">
        {pagesError && !pages ? (
          <div className="viewer__msg">
            <ErrorBanner error={pagesError} onRetry={onRetry} title="Could not load the scanned pages" />
          </div>
        ) : !pages ? (
          <div className="paper-frame">
            <Skeleton width="100%" height={18} radius={6} />
            <div className="paper paper--empty" style={{ width: "100%", aspectRatio: "1 / 1.414" }}>
              <span className="paper__loading skeleton" aria-hidden="true" />
            </div>
          </div>
        ) : pages.length === 0 ? (
          <EmptyState compact illustration="generic" title="No pages" description="This sheet has no rendered pages yet." />
        ) : (
          <>
            {noRegion ? (
              <p className="viewer__note">
                No region is linked to this question, so every page is shown.
                {canEditRegions ? (
                  <>
                    {" "}
                    <Link to={regionsHref} className="inline-link">
                      Draw one in the region editor
                    </Link>
                  </>
                ) : null}
              </p>
            ) : null}
            {shown.map((p) => {
              const pageRegions = paintOrder(regionsByPage.get(p.page_no) ?? []);
              const mine = pageRegions.filter((r) => answerRegionIds.has(r.id));
              return (
                <figure key={p.page_no} className="paper-frame">
                  <figcaption className="paper-caption">
                    <span>Page {p.page_no}</span>
                    {p.flags.map((f) => {
                      const spec = flagSpec(f);
                      const Icon = spec.icon;
                      return (
                        <Badge key={f} size="sm" tone={spec.tone} icon={<Icon size={11} aria-hidden="true" />}>
                          {spec.label}
                        </Badge>
                      );
                    })}
                    {mine.length ? (
                      <span className="sr-only">
                        The answer to question {answer?.question_no} is outlined on this page.
                      </span>
                    ) : null}
                  </figcaption>
                  <Paper
                    submissionId={submissionId}
                    page={p}
                    zoom={zoom}
                    alt={`Scanned page ${p.page_no} of student sheet`}
                  >
                    {(space) => {
                      let first = true;
                      return pageRegions.map((r) => {
                        const st = regionStyle(r, lookup);
                        const isMine = answerRegionIds.has(r.id);
                        const focus = isMine && first;
                        if (focus) first = false;
                        const nearTop = r.bbox[1] / space.h < 0.03;
                        return (
                          <div
                            key={r.id}
                            aria-hidden="true"
                            data-focus-box={focus ? "true" : undefined}
                            className={cx("rbox", "rbox--static", st.dashed && "rbox--dashed", !isMine && "rbox--muted", isMine && "rbox--emph")}
                            style={{ ...boxStyle(r.bbox, space.w, space.h), ["--rc" as string]: isMine ? st.color : "var(--region-unassigned)" }}
                          >
                            <span className={cx("rbox__label", nearTop && "rbox__label--inside")}>
                              {st.label}
                              {isMine && r.source === "manual" ? " · edited" : ""}
                            </span>
                          </div>
                        );
                      });
                    }}
                  </Paper>
                </figure>
              );
            })}
          </>
        )}
      </div>
    </section>
  );
}
