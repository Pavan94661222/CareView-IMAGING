function formatPercent(value) {
  if (!Number.isFinite(value)) return '0.0%'
  return `${(value * 100).toFixed(1)}%`
}

export default function ProbabilityBar({ probabilities }) {
  const normal =
    probabilities && Number.isFinite(probabilities.NORMAL) ? probabilities.NORMAL : 0
  const pneumonia =
    probabilities && Number.isFinite(probabilities.PNEUMONIA) ? probabilities.PNEUMONIA : 0
  const total = normal + pneumonia > 0 ? normal + pneumonia : 1
  const normalWidth = (normal / total) * 100
  const pneumoniaWidth = (pneumonia / total) * 100

  return (
    <div className="prob">
      <div className="prob__legend">
        <span className="prob__legend-item">
          <span className="prob__swatch prob__swatch--normal" aria-hidden="true" />
          NORMAL
          <strong>{formatPercent(normal)}</strong>
        </span>
        <span className="prob__legend-item">
          <span className="prob__swatch prob__swatch--pneumonia" aria-hidden="true" />
          PNEUMONIA
          <strong>{formatPercent(pneumonia)}</strong>
        </span>
      </div>
      <div
        className="prob__track"
        role="img"
        aria-label={`Class probabilities: NORMAL ${formatPercent(normal)}, PNEUMONIA ${formatPercent(
          pneumonia,
        )}`}
      >
        <span className="prob__segment prob__segment--normal" style={{ width: `${normalWidth}%` }} />
        <span
          className="prob__segment prob__segment--pneumonia"
          style={{ width: `${pneumoniaWidth}%` }}
        />
      </div>
    </div>
  )
}
