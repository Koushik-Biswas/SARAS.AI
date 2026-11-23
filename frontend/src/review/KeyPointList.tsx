// Rubric key points of one answer: status (icon + text), awarded / weight, entailment and
// contradiction probabilities, and the student's sentence used as evidence.
import { Quote } from "lucide-react";
import type { KeyPointOut } from "../api/types";
import { StatusPill, cx } from "../components";
import { fmtMarks, pct } from "./flags";

function Prob({ label, value, danger }: { label: string; value: number; danger?: boolean }) {
  const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  return (
    <span className={cx("prob", danger && "prob--danger")}>
      <span className="prob__label">{label}</span>
      <span
        className="prob__track"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(v * 100)}
        aria-valuetext={pct(v)}
      >
        <span className="prob__fill" style={{ width: `${v * 100}%` }} />
      </span>
      <span className="prob__pct num">{pct(v)}</span>
    </span>
  );
}

interface KeyPointListProps {
  points: KeyPointOut[];
  contradictionThreshold: number;
}

export function KeyPointList({ points, contradictionThreshold }: KeyPointListProps) {
  if (points.length === 0) {
    return <p className="subtle kpl__empty">This question has no key points in the rubric, so marks are set by hand.</p>;
  }
  return (
    <ol className="kpl">
      {points.map((kp, i) => (
        <li key={kp.index ?? i} className={cx("kpl__item", `kpl__item--${kp.status}`)}>
          <div className="kpl__head">
            <span className="kpl__index num" aria-hidden="true">
              {i + 1}
            </span>
            <p className="kpl__text">
              <span className="sr-only">Key point {i + 1}: </span>
              {kp.text}
            </p>
            <StatusPill kind="keypoint" status={kp.status} size="sm" />
          </div>
          <div className="kpl__meta">
            <span className="kpl__award num">
              <strong>{fmtMarks(kp.awarded)}</strong>
              <span className="subtle"> / {fmtMarks(kp.weight)}</span>
              <span className="sr-only"> marks awarded</span>
            </span>
            <Prob label="Supports" value={kp.entailment_prob} />
            <Prob label="Contradicts" value={kp.contradiction_prob} danger={kp.contradiction_prob >= contradictionThreshold} />
          </div>
          {kp.evidence_sentence ? (
            <blockquote className="kpl__evidence">
              <Quote size={14} aria-hidden="true" className="kpl__quote-icon" />
              <span className="sr-only">Evidence from the answer: </span>
              <span>“{kp.evidence_sentence}”</span>
            </blockquote>
          ) : (
            <p className="kpl__no-evidence subtle">No supporting sentence found in the answer.</p>
          )}
        </li>
      ))}
    </ol>
  );
}
