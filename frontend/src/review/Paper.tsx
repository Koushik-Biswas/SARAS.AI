// One scanned page: white paper on the sunken canvas, the image fetched as an authenticated
// blob, and an overlay whose children work in page-image pixels (natural size).
import { useEffect, useRef, useState, type HTMLAttributes, type ReactNode, type Ref } from "react";
import { ImageOff, RefreshCw } from "lucide-react";
import type { PageInfo } from "../api/types";
import { Button, cx } from "../components";
import { useInView, usePageImage } from "./hooks";

export interface PageSpace {
  /** Natural page-image width/height in pixels (the coordinate space of every bbox). */
  w: number;
  h: number;
}

interface PaperProps {
  submissionId: string;
  page: PageInfo;
  /** Displayed width relative to the container (1 = fit width). */
  zoom?: number;
  /** Load the 240 px thumbnail instead of the full page. */
  thumb?: boolean;
  /** Fetch only when scrolled near the viewport (default true). */
  lazy?: boolean;
  alt?: string;
  className?: string;
  paperRef?: Ref<HTMLDivElement>;
  /** Extra props for the paper element (pointer handlers, data attributes). */
  paperProps?: HTMLAttributes<HTMLDivElement>;
  /** Overlay content, laid out in percentages of `space`. */
  children?: (space: PageSpace) => ReactNode;
  /** Reports the page-pixel coordinate space whenever it becomes known or changes. */
  onSpace?: (space: PageSpace) => void;
}

export function Paper({ submissionId, page, zoom = 1, thumb = false, lazy = true, alt, className, paperRef, paperProps, children, onSpace }: PaperProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const visible = useInView(hostRef, lazy);
  const img = usePageImage(submissionId, page.page_no, thumb, visible);
  const [natural, setNatural] = useState<PageSpace | null>(null);

  // Thumbnails are scaled copies: their boxes still live in the full page's pixel space.
  const space: PageSpace =
    !thumb && natural ? natural : { w: page.width || 0, h: page.height || 0 };
  const ratio = space.w > 0 && space.h > 0 ? `${space.w} / ${space.h}` : natural ? `${natural.w} / ${natural.h}` : "1 / 1.414";
  const onSpaceRef = useRef(onSpace);
  onSpaceRef.current = onSpace;
  useEffect(() => {
    if (space.w > 0 && space.h > 0) onSpaceRef.current?.({ w: space.w, h: space.h });
  }, [space.w, space.h]);

  return (
    <div ref={hostRef} className={cx("paper-host", className)}>
      <div
        ref={paperRef}
        {...paperProps}
        className={cx("paper", !img.url && "paper--empty", paperProps?.className)}
        style={{ width: `${zoom * 100}%`, aspectRatio: ratio, ...paperProps?.style }}
      >
        {img.url ? (
          <img
            className="paper__img"
            src={img.url}
            alt={alt ?? `Scanned page ${page.page_no}`}
            draggable={false}
            onLoad={(e) => {
              const t = e.currentTarget;
              if (!thumb && t.naturalWidth > 0 && (natural?.w !== t.naturalWidth || natural?.h !== t.naturalHeight)) {
                setNatural({ w: t.naturalWidth, h: t.naturalHeight });
              }
            }}
          />
        ) : img.error ? (
          <div className="paper__error" role="alert">
            <ImageOff size={thumb ? 16 : 22} aria-hidden="true" />
            {!thumb ? <span>Could not load page {page.page_no}</span> : <span className="sr-only">Could not load page {page.page_no}</span>}
            {!thumb ? (
              <Button size="sm" variant="secondary" onClick={img.retry} iconLeft={<RefreshCw size={14} aria-hidden="true" />}>
                Retry
              </Button>
            ) : null}
          </div>
        ) : (
          <span className="paper__loading skeleton" aria-hidden="true" />
        )}
        {children && space.w > 0 && space.h > 0 ? <div className="paper__overlay">{children(space)}</div> : null}
      </div>
    </div>
  );
}
