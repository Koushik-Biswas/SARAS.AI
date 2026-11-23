import { useMemo, useState, type ReactNode } from "react";
import { ArrowDown, ArrowUp, ChevronsUpDown } from "lucide-react";
import { cx } from "./cx";

export interface Column<T> {
  key: string;
  header: ReactNode;
  /** Plain-text header for aria-labels when `header` is not a string. */
  headerText?: string;
  render: (row: T, index: number) => ReactNode;
  /** Enables sorting on this column. */
  sortValue?: (row: T) => string | number | null | undefined;
  align?: "left" | "right" | "center";
  width?: number | string;
  /** Render as a row header (<th scope="row">). */
  rowHeader?: boolean;
  className?: string;
}

type SortDir = "asc" | "desc";

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  caption: string;
  /** Visually show the caption (otherwise screen-reader only). */
  showCaption?: boolean;
  initialSort?: { key: string; dir: SortDir };
  empty?: ReactNode;
  rowClassName?: (row: T) => string | undefined;
  /** Max height for the scroll area (sticky header scrolls inside). */
  maxHeight?: number | string;
  dense?: boolean;
  footer?: ReactNode;
}

function compare(a: unknown, b: unknown): number {
  if (a === b) return 0;
  if (a === null || a === undefined || a === "") return 1; // nulls last
  if (b === null || b === undefined || b === "") return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

/** Table with sticky header, hover rows and optional per-column sorting. */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  caption,
  showCaption,
  initialSort,
  empty,
  rowClassName,
  maxHeight,
  dense,
  footer,
}: DataTableProps<T>) {
  const [sort, setSort] = useState<{ key: string; dir: SortDir } | null>(initialSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const get = col.sortValue;
    const out = [...rows].sort((a, b) => compare(get(a), get(b)));
    if (sort.dir === "desc") {
      // keep nulls last when descending
      out.sort((a, b) => {
        const va = get(a);
        const vb = get(b);
        const na = va === null || va === undefined || va === "";
        const nb = vb === null || vb === undefined || vb === "";
        if (na || nb) return na === nb ? 0 : na ? 1 : -1;
        return -compare(va, vb);
      });
    }
    return out;
  }, [rows, sort, columns]);

  const toggle = (key: string) =>
    setSort((s) => (s && s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }));

  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table className={cx("table", dense && "table--dense")}>
        <caption className={showCaption ? "table__caption" : "sr-only"}>{caption}</caption>
        <thead>
          <tr>
            {columns.map((c) => {
              const active = sort?.key === c.key;
              const ariaSort = active ? (sort!.dir === "asc" ? "ascending" : "descending") : c.sortValue ? "none" : undefined;
              return (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={ariaSort}
                  className={cx(c.align && `align-${c.align}`, c.className)}
                  style={c.width !== undefined ? { width: c.width } : undefined}
                >
                  {c.sortValue ? (
                    <button type="button" className={cx("th-sort", active && "is-active")} onClick={() => toggle(c.key)}>
                      <span>{c.header}</span>
                      {active ? (
                        sort!.dir === "asc" ? (
                          <ArrowUp size={13} aria-hidden="true" />
                        ) : (
                          <ArrowDown size={13} aria-hidden="true" />
                        )
                      ) : (
                        <ChevronsUpDown size={13} aria-hidden="true" className="th-sort__idle" />
                      )}
                    </button>
                  ) : (
                    c.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr className="table__empty-row">
              <td colSpan={columns.length}>{empty ?? <span className="subtle">Nothing to show.</span>}</td>
            </tr>
          ) : (
            sorted.map((row, i) => (
              <tr key={rowKey(row)} className={rowClassName?.(row)}>
                {columns.map((c) =>
                  c.rowHeader ? (
                    <th key={c.key} scope="row" className={cx(c.align && `align-${c.align}`, c.className)}>
                      {c.render(row, i)}
                    </th>
                  ) : (
                    <td key={c.key} className={cx(c.align && `align-${c.align}`, c.className)}>
                      {c.render(row, i)}
                    </td>
                  ),
                )}
              </tr>
            ))
          )}
        </tbody>
        {footer ? <tfoot>{footer}</tfoot> : null}
      </table>
    </div>
  );
}
