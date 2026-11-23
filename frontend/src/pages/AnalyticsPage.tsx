import { useState, type ReactNode } from "react";
import { ArrowDown, ArrowUp, ChartColumn, CircleCheck, Flag, Gauge, Pencil, Sigma, Target, TriangleExclamationPoint, Users } from "lucide-react";
import * as api from "../api/endpoints";
import type { AnalyticsOut, QuestionAnalytics } from "../api/types";
import { Badge, Banner, ButtonLink, Card, EmptyState, ErrorBanner, Segmented, Skeleton, StatTile, Tooltip, cx } from "../components";
import { useAsync } from "../hooks/useAsync";
import { useExam } from "./ExamLayout";

const RELIABLE_SHEETS = 5;
const pct = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? "-" : `${(v * 100).toFixed(digits)}%`);
const fmt = (v: number | null | undefined) => (v === null || v === undefined ? "-" : String(Math.round(v * 100) / 100));
const humanFlag = (f: string) => f.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

type View = "chart" | "table";

function ChartCard({ title, subtitle, view, onView, children }: { title: string; subtitle?: string; view: View; onView: (v: View) => void; children: ReactNode }) {
  return (
    <Card
      title={title}
      subtitle={subtitle}
      actions={
        <Segmented<View>
          size="sm"
          label={`${title}: display`}
          value={view}
          onChange={onView}
          options={[
            { value: "chart", label: "Chart" },
            { value: "table", label: "Table" },
          ]}
        />
      }
    >
      {children}
    </Card>
  );
}

/** Vertical column chart of the 10-bucket mark distribution. */
function DistributionChart({ data }: { data: AnalyticsOut }) {
  const max = Math.max(1, ...data.distribution.map((b) => b.count));
  const ticks = max <= 4 ? Array.from({ length: max + 1 }, (_, i) => i) : [0, Math.round(max / 2), max];
  return (
    <div className="vchart" role="img" aria-label={`Mark distribution across ${data.sheets} sheets. ${data.distribution.map((b) => `${b.from_pct}-${b.to_pct}%: ${b.count}`).join("; ")}`}>
      <div className="vchart__plot">
        <div className="vchart__grid" aria-hidden="true">
          {ticks
            .slice()
            .reverse()
            .map((t) => (
              <div key={t} className="vchart__gridline" style={{ bottom: `${(t / max) * 100}%` }}>
                <span className="vchart__tick num">{t}</span>
              </div>
            ))}
        </div>
        <div className="vchart__bars">
          {data.distribution.map((b) => {
            const h = (b.count / max) * 100;
            const label = `${b.from_pct}-${b.to_pct}%: ${b.count} sheet${b.count === 1 ? "" : "s"}`;
            return (
              <Tooltip key={b.from_pct} content={label} className="vchart__slot">
                <span className="vchart__hit" tabIndex={0} aria-label={label}>
                  {b.count > 0 ? <span className="vchart__value num">{b.count}</span> : null}
                  <span className={cx("vchart__bar", b.count === 0 && "is-zero")} style={{ height: b.count ? `${h}%` : undefined }} />
                </span>
              </Tooltip>
            );
          })}
        </div>
      </div>
      <div className="vchart__axis" aria-hidden="true">
        {data.distribution.map((b) => (
          <span key={b.from_pct} className="num">
            {b.from_pct}
            {b.to_pct === 100 ? "-100" : ""}
          </span>
        ))}
      </div>
      <p className="vchart__caption">Share of total marks (%) · number of sheets</p>
    </div>
  );
}

function DistributionTable({ data }: { data: AnalyticsOut }) {
  return (
    <table className="table table--dense">
      <caption className="sr-only">Mark distribution</caption>
      <thead>
        <tr>
          <th scope="col">Score band</th>
          <th scope="col" className="align-right">
            Sheets
          </th>
        </tr>
      </thead>
      <tbody>
        {data.distribution.map((b) => (
          <tr key={b.from_pct}>
            <th scope="row" className="num">
              {b.from_pct}-{b.to_pct}%
            </th>
            <td className="align-right num">{b.count}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Horizontal bars: average % per question, hardest questions emphasised + labelled. */
function QuestionBars({ questions, hardest }: { questions: QuestionAnalytics[]; hardest: Set<number> }) {
  return (
    <ul className="hbars">
      {questions.map((qa) => {
        const p = qa.average_pct ?? 0;
        const isHard = hardest.has(qa.question_no);
        const label = `Q${qa.question_no}: average ${qa.average_pct ?? "-"}% (${fmt(qa.average)} of ${qa.max_marks})${qa.blank ? `, ${qa.blank} blank` : ""}`;
        return (
          <li key={qa.question_no} className={cx("hbars__row", isHard && "is-hard")}>
            <span className="hbars__label">
              <strong className="num">Q{qa.question_no}</strong>
              <span className="subtle">{qa.type}</span>
            </span>
            <Tooltip content={label} className="hbars__track-host">
              <span className="hbars__track" tabIndex={0} aria-label={label}>
                <span className="hbars__fill" style={{ width: `${Math.max(0, Math.min(100, p))}%` }} />
              </span>
            </Tooltip>
            <span className="hbars__value num">
              {qa.average_pct === null ? "-" : `${qa.average_pct}%`}
              <span className="subtle"> · {fmt(qa.average)}/{qa.max_marks}</span>
            </span>
            <span className="hbars__tag">
              {isHard ? (
                <Badge size="sm" tone="warning" icon={<TriangleExclamationPoint size={12} aria-hidden="true" />}>
                  Hardest
                </Badge>
              ) : null}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function QuestionTable({ questions, hardest }: { questions: QuestionAnalytics[]; hardest: Set<number> }) {
  return (
    <table className="table table--dense">
      <caption className="sr-only">Average marks per question</caption>
      <thead>
        <tr>
          <th scope="col">Question</th>
          <th scope="col">Type</th>
          <th scope="col" className="align-right">
            Average
          </th>
          <th scope="col" className="align-right">
            Average %
          </th>
          <th scope="col" className="align-right">
            Blank
          </th>
        </tr>
      </thead>
      <tbody>
        {questions.map((qa) => (
          <tr key={qa.question_no}>
            <th scope="row">
              Q{qa.question_no} {hardest.has(qa.question_no) ? <Badge size="sm" tone="warning">Hardest</Badge> : null}
            </th>
            <td>{qa.type}</td>
            <td className="align-right num">
              {fmt(qa.average)} / {qa.max_marks}
            </td>
            <td className="align-right num">{qa.average_pct ?? "-"}%</td>
            <td className="align-right num">{qa.blank}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function AnalyticsPage() {
  const { exam } = useExam();
  const q = useAsync(() => api.getAnalytics(exam.id), [exam.id]);
  const [distView, setDistView] = useState<View>("chart");
  const [qView, setQView] = useState<View>("chart");
  const data = q.data;

  if (q.loading && !data) {
    return (
      <div className="stack">
        <div className="stat-grid stat-grid--wide">
          {Array.from({ length: 4 }, (_, i) => (
            <StatTile key={i} label="Loading" value="" loading />
          ))}
        </div>
        <Skeleton height={260} radius={14} />
      </div>
    );
  }
  if (q.error && !data) return <ErrorBanner error={q.error} onRetry={() => q.reload()} title="Could not load analytics" />;
  if (!data || data.sheets === 0) {
    return (
      <Card>
        <EmptyState
          illustration="chart"
          title="Analytics appear after the first sheet is processed"
          description="Upload and process at least one answer sheet to see score distributions, per-question difficulty and the most-missed key points."
          action={
            <ButtonLink to={`/exams/${exam.id}/upload`} variant="primary">
              Upload sheets
            </ButtonLink>
          }
        />
      </Card>
    );
  }

  const hardest = new Set(data.hardest);
  const missed = data.questions.filter((qa) => qa.most_missed.length > 0);
  const maxFlag = Math.max(1, ...data.flags.map(([, n]) => n));
  const of = (v: number | null) => (v === null ? "-" : <>{fmt(v)}<small>/ {fmt(data.max_total)}</small></>);

  return (
    <div className="stack">
      {exam.stats.approved < RELIABLE_SHEETS ? (
        <Banner tone="info" title="Early numbers">
          Based on {data.sheets} processed sheet{data.sheets === 1 ? "" : "s"} ({exam.stats.approved} approved). Statistics become meaningful once at least {RELIABLE_SHEETS} sheets are approved.
        </Banner>
      ) : null}

      <section className="stat-grid stat-grid--wide" aria-label="Score summary">
        <StatTile label="Sheets" value={data.sheets} icon={<Users size={18} />} sub="Processed, in review or approved" />
        <StatTile label="Average" value={of(data.average)} icon={<Sigma size={18} />} tone="info" sub={data.stdev !== null ? `Std. dev. ${fmt(data.stdev)}` : undefined} />
        <StatTile label="Median" value={of(data.median)} icon={<Target size={18} />} tone="info" />
        <StatTile label="Highest" value={of(data.highest)} icon={<ArrowUp size={18} />} tone="success" />
        <StatTile label="Lowest" value={of(data.lowest)} icon={<ArrowDown size={18} />} tone="warning" />
        <StatTile
          label="Auto-accepted"
          value={pct(data.auto_accepted_share)}
          icon={<CircleCheck size={18} />}
          tone="success"
          sub={data.mean_confidence !== null ? `Mean confidence ${pct(data.mean_confidence)}` : "Share of graded answers"}
        />
        <StatTile
          label="Override rate"
          value={pct(data.override_rate)}
          icon={<Pencil size={18} />}
          tone="neutral"
          sub={data.mean_abs_override !== null ? `Mean change ${fmt(data.mean_abs_override)} marks` : "Reviewed answers with changed marks"}
        />
        <StatTile label="Confidence" value={pct(data.mean_confidence)} icon={<Gauge size={18} />} tone="accent" sub="Mean across graded answers" />
      </section>

      <div className="analytics-grid">
        <ChartCard title="Mark distribution" subtitle="Sheets per 10% band of the total" view={distView} onView={setDistView}>
          {distView === "chart" ? <DistributionChart data={data} /> : <DistributionTable data={data} />}
        </ChartCard>
        <ChartCard title="Average by question" subtitle="Lowest-scoring questions are marked Hardest" view={qView} onView={setQView}>
          {qView === "chart" ? <QuestionBars questions={data.questions} hardest={hardest} /> : <QuestionTable questions={data.questions} hardest={hardest} />}
        </ChartCard>
      </div>

      <div className="analytics-grid">
        <Card title="Most-missed key points" subtitle="Key points most often missed or contradicted">
          {missed.length === 0 ? (
            <EmptyState compact illustration="review" title="Nothing stands out" description="No key point has been missed yet." />
          ) : (
            <div className="missed">
              {missed.map((qa) => (
                <div key={qa.question_no} className="missed__group">
                  <p className="missed__q">
                    <strong className="num">Q{qa.question_no}</strong>
                    {hardest.has(qa.question_no) ? (
                      <Badge size="sm" tone="warning" icon={<TriangleExclamationPoint size={12} aria-hidden="true" />}>
                        Hardest
                      </Badge>
                    ) : null}
                  </p>
                  <ol className="missed__list">
                    {qa.most_missed.map(([text, n]) => (
                      <li key={text}>
                        <span className="grow">{text}</span>
                        <span className="missed__count num">
                          {n} sheet{n === 1 ? "" : "s"}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>
          )}
        </Card>
        <Card title="Review flags" subtitle="Why answers were routed to review">
          {data.flags.length === 0 ? (
            <EmptyState compact icon={<Flag size={28} aria-hidden="true" />} title="No flags raised" description="Every answer was graded with high confidence." />
          ) : (
            <ul className="flagbars">
              {data.flags.map(([flag, n]) => (
                <li key={flag}>
                  <span className="flagbars__label">{humanFlag(flag)}</span>
                  <span className="flagbars__track" aria-hidden="true">
                    <span className="flagbars__fill" style={{ width: `${(n / maxFlag) * 100}%` }} />
                  </span>
                  <span className="flagbars__value num">{n}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
      <p className="legend-note">
        <ChartColumn size={14} aria-hidden="true" />
        <span>Marks use reviewer-final values where available, otherwise the AI marks.</span>
      </p>
    </div>
  );
}
