const STATUS = {
  checking: { label: 'Checking backend…', className: 'status--checking' },
  ready: { label: 'Model ready', className: 'status--ready' },
  not_trained: { label: 'Model not trained', className: 'status--warn' },
  offline: { label: 'Backend offline', className: 'status--offline' },
}

function LungMark() {
  return (
    <svg viewBox="0 0 32 32" width="22" height="22" aria-hidden="true" focusable="false">
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M16 5v22" />
        <path d="M16 10.5c-4 0-6.5 3-6.5 7.5 0 4 2 6.5 4 6.5s2.5-2 2.5-5.5" />
        <path d="M16 10.5c4 0 6.5 3 6.5 7.5 0 4-2 6.5-4 6.5s-2.5-2-2.5-5.5" />
      </g>
    </svg>
  )
}

export default function Header({ backend }) {
  const status = STATUS[backend.state] || STATUS.checking

  return (
    <header className="site-header">
      <div className="topbar">
        <div className="brand">
          <span className="brand__mark">
            <LungMark />
          </span>
          <span className="brand__text">Pneumonia Detection System</span>
        </div>

        <div className={`status ${status.className}`} role="status" title={backend.detail || undefined}>
          <span className="status__dot" aria-hidden="true" />
          <span className="status__label">{status.label}</span>
        </div>
      </div>

      <div className="hero">
        <p className="eyebrow">Chest radiograph analysis</p>
        <h1 className="hero__title">Pneumonia Detection System</h1>
        <p className="hero__desc">
          An automated research prototype that applies convolutional neural networks and
          transfer learning to classify chest X-rays as <strong>NORMAL</strong> or{' '}
          <strong>PNEUMONIA</strong>, returning a calibrated confidence score with the full
          probability breakdown.
        </p>

        {backend.state === 'not_trained' && (
          <p className="hero__hint">
            The model has not been trained yet. Generate weights with{' '}
            <code>python -m pneumonia.train</code>.
          </p>
        )}
        {backend.state === 'offline' && (
          <p className="hero__hint">
            The backend could not be reached. Start it with{' '}
            <code>uvicorn backend.main:app --port 8000</code>.
          </p>
        )}
      </div>
    </header>
  )
}
