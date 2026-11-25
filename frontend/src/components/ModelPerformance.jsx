import { useState } from 'react'
import { fetchMetrics } from '../api.js'

const TILES = [
  ['accuracy', 'Accuracy'],
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['f1', 'F1 score'],
  ['roc_auc', 'ROC-AUC'],
  ['specificity', 'Specificity'],
  ['balanced_accuracy', 'Balanced accuracy'],
  ['average_precision', 'Avg. precision'],
]

const FINE_TUNING_TILES = [
  ['accuracy', 'Fine-tuning accuracy'],
  ['precision', 'Training precision'],
  ['recall', 'Training recall'],
  ['f1', 'Training F1 score'],
  ['roc_auc', 'Training ROC-AUC'],
]

function formatMetric(value) {
  if (!Number.isFinite(value)) return '—'
  if (value <= 1) return `${(value * 100).toFixed(1)}%`
  return value.toFixed(2)
}

export default function ModelPerformance() {
  const [open, setOpen] = useState(false)
  const [status, setStatus] = useState('idle')
  const [report, setReport] = useState(null)
  const [message, setMessage] = useState(null)
  const tiles = report?.evaluation_type === 'fine_tuning_set' ? FINE_TUNING_TILES : TILES

  const toggle = async () => {
    const next = !open
    setOpen(next)
    if (!next || status !== 'idle') return

    setStatus('loading')
    try {
      const data = await fetchMetrics()
      if (data && data.available && data.report) {
        setReport(data.report)
        setStatus('ready')
      } else {
        setMessage((data && data.detail) || 'No evaluation report is available yet.')
        setStatus('unavailable')
      }
    } catch (error) {
      setMessage(error.message || 'Could not load model metrics.')
      setStatus('error')
    }
  }

  return (
    <section className="card performance">
      <button type="button" className="performance__toggle" onClick={toggle} aria-expanded={open}>
        <span className="performance__toggle-text">
          <span className="eyebrow">Model metrics</span>
          <span className="performance__title">Model performance</span>
        </span>
        <span
          className={`performance__chevron${open ? ' performance__chevron--open' : ''}`}
          aria-hidden="true"
        >
          <svg viewBox="0 0 20 20" width="18" height="18" focusable="false">
            <path
              d="M6 8l4 4 4-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>

      {open && (
        <div className="performance__body" aria-live="polite">
          {status === 'loading' && <p className="performance__message">Loading metrics…</p>}
          {status === 'error' && (
            <p className="performance__message performance__message--error">{message}</p>
          )}
          {status === 'unavailable' && <p className="performance__message">{message}</p>}

          {status === 'ready' && report && (
            <>
              {report.evaluation_type === 'fine_tuning_set' && (
                <p className="performance__note">
                  These {report.n_samples ?? '—'} labeled images were used to fine-tune this model.
                  The metrics below are training-set performance, not an independent test result.
                </p>
              )}

              <div className="tiles">
                {tiles.map(([key, label]) => (
                  <div className="tile" key={key}>
                    <span className="tile__label">{label}</span>
                    <span className="tile__value">{formatMetric(report[key])}</span>
                  </div>
                ))}
              </div>

              {report.confusion_matrix && (
                <div className="confusion">
                  <span className="eyebrow">Confusion matrix</span>
                  <div className="tiles tiles--matrix">
                    <div className="tile">
                      <span className="tile__label">True negatives</span>
                      <span className="tile__value">{report.confusion_matrix.tn ?? '—'}</span>
                    </div>
                    <div className="tile">
                      <span className="tile__label">False positives</span>
                      <span className="tile__value">{report.confusion_matrix.fp ?? '—'}</span>
                    </div>
                    <div className="tile">
                      <span className="tile__label">False negatives</span>
                      <span className="tile__value">{report.confusion_matrix.fn ?? '—'}</span>
                    </div>
                    <div className="tile">
                      <span className="tile__label">True positives</span>
                      <span className="tile__value">{report.confusion_matrix.tp ?? '—'}</span>
                    </div>
                  </div>
                </div>
              )}

              <p className="performance__note">
                {report.evaluation_type === 'fine_tuning_set'
                  ? `Computed on the fine-tuning set at threshold ${
                      Number.isFinite(report.threshold) ? report.threshold.toFixed(2) : '—'
                    }.`
                  : `Evaluated on ${report.n_samples ?? '—'} held-out samples at a decision threshold of ${
                      Number.isFinite(report.threshold) ? report.threshold.toFixed(2) : '—'
                    }.`}
              </p>
            </>
          )}
        </div>
      )}
    </section>
  )
}
