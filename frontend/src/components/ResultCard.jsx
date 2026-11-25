import ProbabilityBar from './ProbabilityBar.jsx'

const SEVERITY_LABELS = {
  low: 'Low severity',
  moderate: 'Moderate severity',
  high: 'High severity',
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function formatMs(value) {
  if (!Number.isFinite(value)) return '—'
  return `${Math.round(value)} ms`
}

export default function ResultCard({ result, onReset }) {
  const isPneumonia = Boolean(result.is_pneumonia)
  const outcome = isPneumonia ? 'pneumonia' : 'normal'
  const severity = typeof result.severity === 'string' ? result.severity.toLowerCase() : ''

  return (
    <article className={`card result result--${outcome}`} aria-live="polite">
      <div className="result__header">
        <span className="eyebrow">Prediction</span>
        {severity && (
          <span className={`badge badge--${severity}`}>
            {SEVERITY_LABELS[severity] || severity}
          </span>
        )}
      </div>

      <div className="result__verdict">
        <p className="result__label">{result.label || result.prediction}</p>
        <p className="result__confidence">
          <span className="result__confidence-value">{formatPercent(result.confidence)}</span>
          <span className="result__confidence-caption">confidence</span>
        </p>
      </div>

      <ProbabilityBar probabilities={result.probabilities} />

      <dl className="result__stats">
        <div className="stat">
          <dt className="stat__label">Decision threshold</dt>
          <dd className="stat__value">
            {Number.isFinite(result.threshold) ? result.threshold.toFixed(2) : '—'}
          </dd>
        </div>
        <div className="stat">
          <dt className="stat__label">P(PNEUMONIA)</dt>
          <dd className="stat__value">{formatPercent(result.probability)}</dd>
        </div>
        <div className="stat">
          <dt className="stat__label">Processing time</dt>
          <dd className="stat__value">{formatMs(result.processing_ms)}</dd>
        </div>
        <div className="stat">
          <dt className="stat__label">Inference time</dt>
          <dd className="stat__value">{formatMs(result.inference_ms)}</dd>
        </div>
      </dl>

      {result.disclaimer && <p className="result__disclaimer">{result.disclaimer}</p>}

      <button type="button" className="button button--primary button--block" onClick={onReset}>
        Analyze another image
      </button>
    </article>
  )
}
