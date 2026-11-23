# SARAS AI design system

Hand-built CSS, no UI kit. Two themes (light + dark) driven entirely by CSS custom properties.

## Files

| File | Purpose |
|---|---|
| `tokens.css` | **The only file allowed to contain colour values.** `:root` = light, `[data-theme="dark"]` = dark. |
| `base.css` | Reset, typography (Inter Variable), focus ring, keyframes, `prefers-reduced-motion`, utilities (`.row`, `.stack`, `.num`, `.sr-only`, `.truncate`, `.muted`, `.subtle`). |
| `components.css` | Styles for everything in `src/components`. |
| `shell.css` | Sidebar, blurred top bar, breadcrumbs, user menu, `.page`, `.page-header`, splash. |
| `pages.css` | Screen-specific layouts (auth, dashboard, exam header, rubric, upload, results, analytics charts). |

Rule: in TSX/CSS use `var(--token)` only. Check with
`rg "#[0-9a-fA-F]{3,8}\b|rgba?\(" src --glob '!**/tokens.css'` (must print nothing).

## Tokens (semantic)

- Surfaces: `--bg`, `--bg-subtle`, `--surface`, `--surface-raised`, `--surface-sunken`, `--surface-hover`, `--surface-translucent` (blurred bars).
- Lines: `--border` (decorative hairline), `--border-strong` (control outlines, at least 3:1).
- Text: `--text`, `--text-muted`, `--text-subtle` (at least 4.5:1 on every surface), `--text-inverse`.
- Accent: `--accent`, `--accent-hover`, `--accent-active`, `--accent-soft`, `--accent-soft-hover`, `--accent-text` (accent-coloured text/links), `--accent-contrast` (text on `--accent`), `--accent-gradient`.
- Status: `--success|warning|danger|info` + `-soft` (pill background) + `-text` (pill text, at least 5.5:1 on soft). `--neutral-soft`, `--neutral-text`.
- Interaction: `--focus-ring`, `--shadow-focus`, `--overlay`, `--selection`, `--skeleton-base/-shine`.
- Elevation: `--shadow-sm|md|lg` (dark theme relies on lighter surfaces + hairlines instead).
- Charts: `--chart-bar`, `--chart-bar-hover`, `--chart-bar-strong` (highlight), `--chart-track`, `--chart-grid`.
- **Page viewer (same in both themes, paper is always white):** `--paper`, `--paper-shadow`, `--region-auto`, `--region-manual`, `--region-selected`, `--region-blank`, `--region-flagged`, `--region-label-text`, `--region-fill`, `--region-fill-selected`. Use 2px outlines plus a label chip (`background: var(--region-*)`, `color: var(--region-label-text)`) on a `--surface-sunken` canvas.
- Shape: `--radius-xs(4) sm(6) md(10) lg(14) xl(20) full`. Spacing `--space-1..16` (4px grid). Type `--text-xs(12) sm(13) md(14) lg(16) xl(18) 2xl(22) 3xl(28) 4xl(34)`.
- Motion: `--dur-fast(150) base(180) slow(220)`, `--ease-out`. Everything is disabled under `prefers-reduced-motion`.
- Layering: `--z-sticky < topbar < sidebar < dropdown < overlay < modal < toast < tooltip`.

## Theme

`src/theme/ThemeProvider.tsx` exposes `useTheme()` -> `{ mode: "light"|"dark"|"system", resolved, setMode, cycle }`.
The choice is persisted in `localStorage["saras-theme"]`, and "system" follows `prefers-color-scheme` live.
`index.html` contains an inline script that sets `data-theme` before first paint (no flash).
The provider also sets `color-scheme` and `<meta name="theme-color">` from `--bg`.
Toggles: `<ThemeToggleButton/>` (top bar, cycles), `<ThemeSegmented/>` (user menu, login corner).

## Components (`import { ... } from "../components"`)

- `Button` (`variant` primary|secondary|ghost|danger|subtle, `size` sm|md|lg, `loading`, `iconLeft`/`iconRight`), `IconButton` (`label` is required: it is the accessible name), `ButtonLink` (router link styled as a button).
- `Field` (label + hint + error wiring for one child control), `Input` (`iconLeft`, `endAdornment`), `NumberInput` (`value: number|null`, `onValueChange`, `decimals` defaults to 2), `Textarea`, `Select`.
- `Segmented` (radio group), `NavTabs` (route pill tabs, animated indicator, `badge`, `alsoMatch`), `Tabs` (in-page ARIA tablist).
- `Card` (`title`, `subtitle`, `actions`, `footer`, `flush` for edge-to-edge tables), `StatTile`, `ProgressBar` (indeterminate when `value` is null), `ConfidenceMeter` (bar + % + High/Medium/Low word), `Skeleton`, `SkeletonTable`.
- `Badge`, `CountBadge`, `StatusPill kind="submission"|"answer"|"keypoint"` (icon + text + tone, never colour alone); `statusSpec()` / `statusLabel()`.
- `Spinner` (immediate, for buttons) and `DelayedSpinner` (appears only after 300 ms, via `useDelayedFlag`).
- `ToastProvider` / `useToast()` -> `success|error|info(title, message?)`. Toasts are polite live-region announcements, stacked bottom-right, and pause on hover.
- `Modal` (portal, focus trap, Esc, focus return, `dismissible`), `ConfirmDialog` (`tone="danger"`, optional required `reasonLabel`: use it for reopen and override reasons).
- `DropdownMenu` + `MenuItem` / `MenuSeparator` / `MenuLabel` (arrow keys, Esc, outside click).
- `Tooltip` (hover + focus; the trigger must be focusable), `EmptyState` (`illustration` exams|upload|review|chart|results|generic), `Banner`, `ErrorBanner` (`onRetry`), `Kbd`, `DataTable` (sticky header, `sortValue` per column, `rowHeader`), `Avatar`, `Logo` / `LogoMark`.

## Hooks

`useAsync(fn, deps)` -> `{data, error, loading, fetching, reload(silent?), setData}`; `usePolling(tick, ms, enabled)` (no overlap, pauses when the tab is hidden); `useDelayedFlag`; `useDocumentTitle`; `useLocalStorage`.
Shell: `useExamIndex()` (sidebar/dashboard exam list), `useBreadcrumbs(level, crumbs)` (level 1 = exam + tab, set by the exam layout; level 2 = sub-page such as a student sheet), and `useExam()` (current exam inside `/exams/:examId/*`; `reload()` refreshes tab badges).

## Pages owned by the next agent

`src/pages/ReviewPage.tsx` (props `{ examId, submissionId? }`; without `submissionId` it renders `<ReviewQueue/>`) and `src/pages/RegionEditorPage.tsx` (props `{ examId, submissionId }`) are placeholders.
All API calls they need already exist in `src/api/endpoints.ts`:
`listAnswers`, `getAnswer`, `reviewAnswer`, `flagAnswer`, `approveSubmission`, `reopenSubmission`, `getPages`, `getPageImage` (Blob -> `URL.createObjectURL`), `listRegions`, `createRegion`, `patchRegion`, `undoRegions`, `listAudit`, `getCapabilities` (thresholds for `ConfidenceMeter` `tau`/`tauLow`).
