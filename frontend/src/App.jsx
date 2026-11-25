import { useCallback, useEffect, useState } from 'react'
import AuthScreen from './components/AuthScreen.jsx'
import ModelPerformance from './components/ModelPerformance.jsx'
import ProbabilityBar from './components/ProbabilityBar.jsx'
import {
  changePassword,
  checkHealth,
  createAnalysis,
  deleteAnalysis,
  deletePatient,
  downloadReport,
  fetchAnalyses,
  fetchAnalysis,
  fetchDashboard,
  fetchPatient,
  fetchPatients,
  getCurrentUser,
  logoutAccount,
  updateProfile,
} from './api.js'

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Overview', icon: '⌂' },
  { id: 'new', label: 'New analysis', icon: '+' },
  { id: 'history', label: 'Analysis history', icon: '▤' },
  { id: 'patients', label: 'Patients', icon: '♧' },
  { id: 'settings', label: 'Settings', icon: '⚙' },
]

function dateLabel(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function Percent({ value }) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function Classification({ value }) {
  const normal = value === 'NORMAL'
  return <span className={`portal-badge ${normal ? 'portal-badge--normal' : 'portal-badge--pneumonia'}`}>{value || 'Unknown'}</span>
}

function ScoreMeter({ label, value, tone = 'teal' }) {
  const score = Math.max(0, Math.min(1, Number(value) || 0))
  return (
    <div className={`portal-score portal-score--${tone}`}>
      <div className="portal-score__heading">
        <span>{label}</span>
        <strong>{Percent({ value: score })}</strong>
      </div>
      <div
        className="portal-score__track"
        role="progressbar"
        aria-label={`${label} model score`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(score * 100)}
      >
        <span style={{ width: `${score * 100}%` }} />
      </div>
    </div>
  )
}

function Brand() {
  return (
    <div className="portal-brand">
      <span className="portal-brand__mark" aria-hidden="true">+</span>
      <span>CareView <small>IMAGING</small></span>
    </div>
  )
}

function Topbar({ user, backendState, onLogout, onNavigate }) {
  return (
    <header className="portal-topbar">
      <button className="portal-mobile-menu" onClick={() => onNavigate('dashboard')} aria-label="Go to overview">☰</button>
      <Brand />
      <div className="portal-topbar__right">
        <span className={`portal-status portal-status--${backendState}`}>
          <i /> {backendState === 'ready' ? 'Model ready' : backendState === 'offline' ? 'API offline' : 'Connecting'}
        </span>
        <span className="portal-user-avatar" aria-hidden="true">{(user.display_name || user.email || 'U').slice(0, 1).toUpperCase()}</span>
        <span className="portal-user-name">{user.display_name}</span>
        <button className="portal-button portal-button--quiet" onClick={onLogout}>Sign out</button>
      </div>
    </header>
  )
}

function Sidebar({ page, onNavigate, user }) {
  return (
    <aside className="portal-sidebar">
      <div className="portal-sidebar__section-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        {NAV_ITEMS.map((item) => (
          <button
            type="button"
            key={item.id}
            className={`portal-nav-item${page === item.id ? ' portal-nav-item--active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <span className="portal-nav-item__icon" aria-hidden="true">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>
      <div className="portal-sidebar__bottom">
        <div className="portal-local-note"><span aria-hidden="true">◉</span><div><b>Private workspace</b><small>Records are stored locally</small></div></div>
        <div className="portal-sidebar__account">{user.email}</div>
      </div>
    </aside>
  )
}

function PageTitle({ eyebrow, title, detail, action }) {
  return (
    <div className="portal-page-title">
      <div>
        <p className="portal-eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {detail && <p className="portal-muted">{detail}</p>}
      </div>
      {action}
    </div>
  )
}

function EmptyState({ title, detail, action }) {
  return (
    <div className="portal-empty">
      <span className="portal-empty__icon" aria-hidden="true">✦</span>
      <h3>{title}</h3>
      <p>{detail}</p>
      {action}
    </div>
  )
}

function ModelStatusPanel({ backend }) {
  const model = backend.model
  const label =
    backend.state === 'ready'
      ? 'Model ready'
      : backend.state === 'offline'
        ? 'API offline'
        : backend.state === 'not_trained'
          ? 'Model not trained'
          : 'Connecting'
  return (
    <aside className="portal-hero__model">
      <div className="portal-hero__model-head">
        <span className={`portal-status portal-status--${backend.state}`}><i />{label}</span>
      </div>
      <dl className="portal-hero__model-grid">
        <div><dt>Architecture</dt><dd>{model?.backbone || '—'}</dd></div>
        <div><dt>Input size</dt><dd>{model?.img_size ? `${model.img_size[0]} × ${model.img_size[1]}` : '—'}</dd></div>
        <div><dt>Decision threshold</dt><dd>{Number.isFinite(model?.threshold) ? model.threshold.toFixed(2) : '—'}</dd></div>
        <div><dt>Positive class</dt><dd>{model?.positive_class || 'PNEUMONIA'}</dd></div>
      </dl>
    </aside>
  )
}

function Overview({ summary, backend, user, onNavigate, onOpenAnalysis }) {
  const latest = summary?.recent_analyses || []
  const firstName = (user?.display_name || 'clinician').trim().split(/\s+/)[0]
  return (
    <>
      <section className="portal-hero">
        <div className="portal-hero__content">
          <p className="portal-eyebrow">Workspace overview</p>
          <h1>Welcome back, {firstName}</h1>
          <p className="portal-hero__lead">
            Review recent activity, monitor classification outcomes, and start a new chest image
            analysis — every image is processed locally by your model.
          </p>
          <div className="portal-hero__actions">
            <button className="portal-button portal-button--primary" onClick={() => onNavigate('new')}>＋ New analysis</button>
            <button className="portal-button" onClick={() => onNavigate('history')}>View history</button>
          </div>
        </div>
        <ModelStatusPanel backend={backend} />
      </section>
      <div className="portal-info-strip">
        <span className="portal-info-strip__icon" aria-hidden="true">i</span>
        <p><b>Transparent, local analysis</b><span>Image classifications come from the app's local model.</span></p>
        <span className="portal-source-badge">LOCAL MODEL</span>
      </div>
      {backend.state === 'offline' && <p className="portal-alert portal-alert--error">The API is unavailable. Check that the backend is running on port 8000.</p>}
      {backend.state === 'not_trained' && <p className="portal-alert portal-alert--warning">The image model is not ready. Check the backend model configuration.</p>}
      <div className="portal-stat-grid">
        <div className="portal-stat-card"><span className="portal-stat-card__icon portal-stat-card__icon--blue">▤</span><div><p>Total analyses</p><strong>{summary?.analysis_count ?? '—'}</strong></div><small>Saved in your workspace</small></div>
        <div className="portal-stat-card"><span className="portal-stat-card__icon portal-stat-card__icon--teal">♧</span><div><p>Patients</p><strong>{summary?.patient_count ?? '—'}</strong></div><small>With saved clinical records</small></div>
        <div className="portal-stat-card"><span className="portal-stat-card__icon portal-stat-card__icon--green">✓</span><div><p>Normal classification</p><strong>{summary?.normal_count ?? '—'}</strong></div><small>Model output only</small></div>
        <div className="portal-stat-card"><span className="portal-stat-card__icon portal-stat-card__icon--rose">⌕</span><div><p>Pneumonia classification</p><strong>{summary?.pneumonia_count ?? '—'}</strong></div><small>Requires clinical review</small></div>
      </div>
      <section className="portal-panel">
        <div className="portal-panel__heading"><div><h2>Recent analyses</h2><p>Your latest saved image reviews</p></div><button className="portal-text-button" onClick={() => onNavigate('history')}>View history →</button></div>
        {latest.length ? <AnalysisTable items={latest} onOpen={onOpenAnalysis} /> : <EmptyState title="No analyses yet" detail="Upload a chest image to create your first saved analysis." action={<button className="portal-button portal-button--primary" onClick={() => onNavigate('new')}>Start an analysis</button>} />}
      </section>
      <ModelPerformance />
    </>
  )
}

function AnalysisTable({ items, onOpen, showPatient = true }) {
  return (
    <div className="portal-table-wrap">
      <table className="portal-table">
        <thead><tr><th>Analysis</th>{showPatient && <th>Patient</th>}<th>Classification</th><th>Confidence</th><th>Date</th><th /></tr></thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.id} onClick={() => onOpen(item.id)} className="portal-table__clickable">
              <td><b>{item.image_filename || 'Chest X-ray'}</b><small>{item.id.slice(0, 8).toUpperCase()}</small></td>
              {showPatient && <td>{item.patient_name || item.patient_code || 'Unassigned'}</td>}
              <td><Classification value={item.result?.prediction} /></td>
              <td>{Percent({ value: item.result?.confidence })}</td>
              <td>{dateLabel(item.created_at)}</td>
              <td><button className="portal-icon-button" aria-label="Open analysis">›</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function NewAnalysis({ csrfToken, onCreated }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  function acceptFile(selected) {
    if (!selected) return
    setError('')
    if (selected.size > 10 * 1024 * 1024) {
      if (preview) URL.revokeObjectURL(preview)
      setFile(null)
      setPreview('')
      setError('Choose an image smaller than 10 MB.')
      return
    }
    const acceptedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']
    const acceptedExtensions = /\.(jpe?g|png|bmp|tiff?)$/i
    if (!acceptedTypes.includes(selected.type) && !acceptedExtensions.test(selected.name || '')) {
      if (preview) URL.revokeObjectURL(preview)
      setFile(null)
      setPreview('')
      setError('Choose a supported chest image (JPEG, PNG, BMP or TIFF).')
      return
    }
    if (preview) URL.revokeObjectURL(preview)
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
  }

  function chooseFile(event) {
    acceptFile(event.target.files?.[0])
    event.target.value = ''
  }

  function updateCounter(event) {
    const field = event.target.closest('.portal-context-field')
    const counter = field?.querySelector('.portal-context-field__counter')
    if (counter) counter.textContent = `${event.target.value.length} / ${event.target.maxLength}`
  }

  function dropFile(event) {
    event.preventDefault()
    setDragging(false)
    acceptFile(event.dataTransfer?.files?.[0])
  }

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  async function submit(event) {
    event.preventDefault()
    if (!file) {
      setError('Select a chest X-ray image to continue.')
      return
    }
    setBusy(true)
    setError('')
    const form = new FormData(event.currentTarget)
    form.set('file', file)
    if (!form.get('age')) form.delete('age')
    try {
      onCreated(await createAnalysis(form, csrfToken))
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageTitle eyebrow="New analysis" title="Review a chest image" detail="Upload an image and add optional patient context to save a private report." />
      <div className="portal-steps">
        <span className="portal-step portal-step--active"><i>1</i>Upload image</span>
        <span className="portal-step__line" />
        <span className={`portal-step${file ? ' portal-step--active' : ''}`}><i>2</i>Add context</span>
        <span className="portal-step__line" />
        <span className={`portal-step${busy ? ' portal-step--active' : ''}`}><i>3</i>Analyze &amp; save</span>
      </div>
      <form className="portal-analysis-form" onSubmit={submit}>
        <section className="portal-panel">
          <div className="portal-panel__heading"><div><h2>Chest image</h2><p>JPEG, PNG, BMP or TIFF · maximum 10 MB</p></div></div>
          <label
            className={`portal-dropzone${preview ? ' portal-dropzone--has-image' : ''}${dragging ? ' portal-dropzone--dragging' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={dropFile}
          >
            {preview ? <img src={preview} alt="Selected chest X-ray preview" /> : <span className="portal-dropzone__icon">↑</span>}
            <span><b>{file ? file.name : 'Choose or drop an image'}</b><small>{file ? 'Click to choose a different image' : 'Your file is sent only to this application server'}</small></span>
            <input type="file" accept="image/jpeg,image/png,image/bmp,image/tiff" onChange={chooseFile} />
          </label>
          {file && <button type="button" className="portal-text-button portal-file-remove" onClick={() => { setFile(null); setPreview(''); setError('') }}>Remove selected image</button>}
        </section>
        <section className="portal-panel">
          <div className="portal-panel__heading"><div><h2>Patient information</h2><p>Optional. Helps organize reports; not used to alter the model output.</p></div></div>
          <div className="portal-form-grid">
            <label>Patient ID<input name="patient_code" maxLength={80} placeholder="e.g. PT-1042" /></label>
            <label>Patient name<input name="patient_name" maxLength={120} placeholder="Full name" /></label>
            <label>Age<input name="age" type="number" min="0" max="130" placeholder="Years" /></label>
            <label>Sex<select name="sex" defaultValue=""><option value="">Not provided</option><option>Female</option><option>Male</option><option>Other</option><option>Prefer not to say</option></select></label>
          </div>
        </section>
        <section className="portal-panel portal-context-panel">
          <div className="portal-panel__heading">
            <div className="portal-panel__heading-with-icon">
              <span className="portal-context-icon" aria-hidden="true">♡</span>
              <div>
                <h2>Clinical context</h2>
                <p>Optional notes are saved with the record. They do not affect the classifier's prediction.</p>
              </div>
            </div>
          </div>
          <div className="portal-form-grid portal-form-grid--single">
            <label className="portal-context-field">
              <span className="portal-context-field__head">
                <span>Symptoms</span>
                <small>e.g. fever, cough, shortness of breath</small>
              </span>
              <textarea name="symptoms" rows="3" maxLength={2000} placeholder="Describe relevant symptoms…" onInput={updateCounter} />
              <span className="portal-context-field__counter">0 / 2000</span>
            </label>
            <label className="portal-context-field">
              <span className="portal-context-field__head">
                <span>Relevant clinical history</span>
                <small>chronic conditions, medications, prior imaging</small>
              </span>
              <textarea name="clinical_history" rows="3" maxLength={4000} placeholder="Relevant history, if known…" onInput={updateCounter} />
              <span className="portal-context-field__counter">0 / 4000</span>
            </label>
            <label className="portal-context-field">
              <span className="portal-context-field__head">
                <span>Notes</span>
                <small>private notes for your records</small>
              </span>
              <textarea name="notes" rows="3" maxLength={4000} placeholder="Additional private notes…" onInput={updateCounter} />
              <span className="portal-context-field__counter">0 / 4000</span>
            </label>
          </div>
        </section>
        {error && <p className="portal-alert portal-alert--error" role="alert">{error}</p>}
        <div className="portal-form-actions">
          <p>{busy ? 'Running the local image classifier and saving your report…' : 'Local model only · no Gemini request · not a diagnosis'}</p>
          <button className="portal-button portal-button--primary" disabled={busy || !file}>{busy ? 'Analyzing and saving…' : 'Analyze and save'}</button>
        </div>
        {busy && <div className="portal-progress" role="status"><span />Processing image locally. This may take a few seconds.</div>}
      </form>
    </>
  )
}

function AnalysisDetail({ analysis, onBack, onNew, onDelete, onDownload }) {
  if (!analysis) return <div className="portal-loading">Loading analysis…</div>
  const output = analysis.result || {}
  const explanation = output.explanation || {}
  const source = explanation.source || {
    provider: 'Local image classifier',
    generation_method: 'No Gemini integration is configured for this application.',
    gemini_used: false,
  }
  const scoreBreakdown = explanation.score_breakdown || {}
  const decision = explanation.decision_detail || {}
  const context = explanation.clinical_context || {}
  const scope = explanation.scope || {}
  const technical = explanation.technical_details || {}
  const normalScore = scoreBreakdown.normal_score ?? output.probabilities?.NORMAL ?? 0
  const pneumoniaScore = scoreBreakdown.pneumonia_score ?? output.probabilities?.PNEUMONIA ?? output.probability
  return (
    <>
      <PageTitle
        eyebrow="Saved report"
        title="Detailed analysis"
        detail={`Created ${dateLabel(analysis.created_at)} · ${analysis.image_filename}`}
        action={<button className="portal-button portal-button--quiet" onClick={onBack}>← Back to records</button>}
      />
      <div className="portal-source-line">
        <span className="portal-source-badge"><i aria-hidden="true" />{source.provider || 'Local image classifier'}</span>
        <span className="portal-source-line__method">{source.generation_method || 'Deterministic explanation from model output'}</span>
      </div>
      <div className="portal-detail-grid portal-detail-grid--analysis">
        <section className={`portal-panel portal-result-panel portal-result-panel--${output.prediction === 'PNEUMONIA' ? 'positive' : 'normal'}`}>
          <div className="portal-result-panel__heading">
            <div><p className="portal-eyebrow">Classifier result</p><h2>{output.prediction === 'PNEUMONIA' ? 'Pneumonia class' : 'Normal class'}</h2></div>
            <Classification value={output.prediction} />
          </div>
          <p className="portal-result-summary">{explanation.summary || 'No explanation is available.'}</p>
          <ProbabilityBar probabilities={{ NORMAL: normalScore, PNEUMONIA: pneumoniaScore }} />
          <div className="portal-score-list">
            <ScoreMeter label="Normal class score" value={normalScore} tone="green" />
            <ScoreMeter label="Pneumonia class score" value={pneumoniaScore} tone="rose" />
          </div>
          <div className="portal-confidence-note">
            <b>{Percent({ value: output.confidence })} selected-class score</b>
            <span>{scoreBreakdown.probability_interpretation || 'A model score is not a guarantee of diagnostic accuracy.'}</span>
          </div>
        </section>
        <section className="portal-panel portal-image-panel portal-image-panel--analysis">
          <div className="portal-panel__heading"><div><h2>Uploaded image</h2><p>Original image used for this model run</p></div></div>
          <img src={analysis.image_url} alt={`Saved chest X-ray ${analysis.image_filename}`} />
          <p>{analysis.image_width} × {analysis.image_height}px <span>·</span> {analysis.image_filename}</p>
        </section>
      </div>
      <div className="portal-detail-grid portal-detail-grid--explanation">
        <section className="portal-panel portal-analysis-card">
          <div className="portal-section-icon portal-section-icon--blue" aria-hidden="true">↗</div>
          <p className="portal-eyebrow">How the label was selected</p>
          <h2>Decision breakdown</h2>
          <p>{decision.explanation || explanation.model_output}</p>
          <div className="portal-detail-stats">
            <div><span>Configured cutoff</span><b>{Number(decision.threshold ?? output.threshold ?? 0).toFixed(2)}</b></div>
            <div><span>Distance from cutoff</span><b>{Number(decision.distance_from_threshold ?? Math.abs(Number(output.probability || 0) - Number(output.threshold || 0))).toFixed(3)}</b></div>
          </div>
          <p className="portal-card-footnote">{decision.distance_note || 'The cutoff distance is a model property, not clinical certainty.'}</p>
        </section>
        <section className="portal-panel portal-analysis-card">
          <div className="portal-section-icon portal-section-icon--teal" aria-hidden="true">♡</div>
          <p className="portal-eyebrow">Clinical context</p>
          <h2>What informed this result?</h2>
          <p>{context.note || 'The model classifies the image only; patient context does not alter its prediction.'}</p>
          <div className="portal-context-state">
            <span>{context.recorded || analysis.symptoms || analysis.clinical_history ? 'Context saved with record' : 'No context provided'}</span>
            <b>Not used by image model</b>
          </div>
          {analysis.symptoms && <p className="portal-context-copy"><b>Symptoms:</b> {analysis.symptoms}</p>}
          {analysis.clinical_history && <p className="portal-context-copy"><b>History:</b> {analysis.clinical_history}</p>}
          {analysis.notes && <p className="portal-context-copy"><b>Notes:</b> {analysis.notes}</p>}
        </section>
      </div>
      <section className="portal-panel portal-next-steps">
        <div className="portal-next-steps__title"><div className="portal-section-icon portal-section-icon--gold" aria-hidden="true">✓</div><div><p className="portal-eyebrow">For discussion with a clinician</p><h2>Suggested next steps</h2></div></div>
        <p>{explanation.recommendation || 'Discuss the result with a qualified healthcare professional.'}</p>
        <p className="portal-next-steps__guidance">{explanation.health_improvements || 'Ask your clinician for personalized health guidance.'}</p>
        <div className="portal-safety-note"><b>Important:</b> This research prototype does not diagnose, rule out disease, localize an abnormality, or replace a radiologist. Follow professional medical advice, especially if symptoms are severe or worsening.</div>
      </section>
      <details className="portal-panel portal-technical-details">
        <summary><span><b>Model and processing details</b><small>Transparency about how this result was produced</small></span><span aria-hidden="true">⌄</span></summary>
        <div className="portal-technical-grid">
          <div><span>Model</span><b>{technical.model_name || output.model?.name || 'Pneumonia image classifier'}</b></div>
          <div><span>Architecture</span><b>{technical.architecture || output.model?.backbone || 'Not available'}</b></div>
          <div><span>Input dimensions</span><b>{(technical.input_size || output.model?.img_size || [analysis.image_width, analysis.image_height]).join(' × ')} px</b></div>
          <div><span>Decision threshold</span><b>{Number(technical.decision_threshold ?? output.threshold ?? 0).toFixed(2)}</b></div>
          <div><span>Inference time</span><b>{Math.round(Number(output.inference_ms || 0))} ms</b></div>
          <div><span>Processing time</span><b>{Math.round(Number(output.processing_ms || 0))} ms</b></div>
          <div><span>Explanation source</span><b>{source.provider || 'Local image classifier'}</b></div>
          <div><span>Gemini API request</span><b>{source.gemini_used ? 'Used' : 'Not used'}</b></div>
        </div>
        <p className="portal-technical-note">{scope.note || 'This is a whole-image classifier. It does not generate localized findings or a radiologist report.'}</p>
        <p className="portal-disclaimer">{explanation.disclaimer || output.disclaimer}</p>
      </details>
      <div className="portal-detail-actions">
        <button className="portal-button portal-button--primary" onClick={() => onDownload(analysis.id)}>Download detailed PDF</button>
        <button className="portal-button" onClick={onNew}>＋ New analysis</button>
        <button className="portal-button portal-button--danger" onClick={() => onDelete(analysis.id)}>Delete analysis</button>
      </div>
    </>
  )
}

function History({ items, query, setQuery, prediction, setPrediction, onOpen }) {
  return (
    <>
      <PageTitle eyebrow="Records" title="Analysis history" detail="Search and review saved image analyses." />
      <section className="portal-panel">
        <div className="portal-filter-row">
          <label className="portal-search">⌕ <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search patient, image or ID" /></label>
          <select value={prediction} onChange={(event) => setPrediction(event.target.value)} aria-label="Filter by classification"><option value="">All classifications</option><option value="NORMAL">Normal</option><option value="PNEUMONIA">Pneumonia</option></select>
        </div>
        {items.length ? <AnalysisTable items={items} onOpen={onOpen} /> : <EmptyState title="No matching analyses" detail="Try changing your search or create a new analysis." />}
      </section>
    </>
  )
}

function Patients({ patients, query, setQuery, onOpen, onDelete }) {
  return (
    <>
      <PageTitle eyebrow="Records" title="Patients" detail="Patient records are private to your account." />
      <section className="portal-panel">
        <div className="portal-filter-row"><label className="portal-search">⌕ <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search patients" /></label></div>
        {patients.length ? (
          <div className="portal-table-wrap"><table className="portal-table"><thead><tr><th>Patient</th><th>Patient ID</th><th>Age</th><th>Analyses</th><th>Last activity</th><th /></tr></thead>
            <tbody>{patients.map((patient) => <tr key={patient.id}>
              <td><b>{patient.name || 'Name not provided'}</b><small>{patient.sex || '—'}</small></td>
              <td>{patient.patient_code || '—'}</td><td>{patient.age ?? '—'}</td><td>{patient.analysis_count}</td><td>{dateLabel(patient.last_seen)}</td>
              <td><button className="portal-button portal-button--quiet" onClick={() => onOpen(patient.id)}>View</button><button className="portal-icon-button portal-icon-button--danger" aria-label={`Delete ${patient.name || 'patient'}`} onClick={() => onDelete(patient.id)}>×</button></td>
            </tr>)}</tbody>
          </table></div>
        ) : <EmptyState title="No patient records" detail="Patient records are created when patient information is added to an analysis." />}
      </section>
    </>
  )
}

function PatientDetail({ patient, onBack, onOpenAnalysis }) {
  if (!patient) return <div className="portal-loading">Loading patient…</div>
  return (
    <>
      <PageTitle eyebrow="Patient record" title={patient.name || 'Patient'} detail={`ID: ${patient.patient_code || 'Not provided'} · ${patient.age ?? 'Age not provided'} · ${patient.sex || 'Sex not provided'}`} action={<button className="portal-button portal-button--quiet" onClick={onBack}>← Patients</button>} />
      <section className="portal-panel"><div className="portal-panel__heading"><div><h2>Related analyses</h2><p>{patient.analysis_count} saved analysis{patient.analysis_count === 1 ? '' : 'es'}</p></div></div>
        {patient.analyses?.length ? <AnalysisTable items={patient.analyses} onOpen={onOpenAnalysis} showPatient={false} /> : <EmptyState title="No analyses linked" detail="New analyses using this patient ID will appear here." />}
      </section>
    </>
  )
}

function Settings({ user, csrfToken, onUserUpdated, onCsrfUpdated }) {
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  async function saveProfile(event) {
    event.preventDefault()
    setBusy(true); setError(''); setMessage('')
    try {
      const response = await updateProfile(new FormData(event.currentTarget).get('display_name'), csrfToken)
      onUserUpdated(response.user)
      setMessage('Profile updated.')
    } catch (requestError) { setError(requestError.message) } finally { setBusy(false) }
  }
  async function savePassword(event) {
    event.preventDefault()
    setBusy(true); setError(''); setMessage('')
    const formElement = event.currentTarget
    const values = Object.fromEntries(new FormData(event.currentTarget).entries())
    try {
      const response = await changePassword(values.current_password, values.new_password, csrfToken)
      onUserUpdated(response.user)
      onCsrfUpdated(response.csrf_token)
      formElement.reset()
      setMessage('Password updated. Other active sessions were signed out.')
    } catch (requestError) { setError(requestError.message) } finally { setBusy(false) }
  }
  return (
    <>
      <PageTitle eyebrow="Workspace" title="Settings" detail="Manage your account and local application security." />
      {error && <p className="portal-alert portal-alert--error">{error}</p>}{message && <p className="portal-alert portal-alert--success">{message}</p>}
      <section className="portal-panel portal-settings"><h2>Profile</h2><p>Update how your account is identified.</p>
        <form className="portal-form portal-settings-form" onSubmit={saveProfile}><label>Email address<input value={user.email} disabled /></label><label>Display name<input name="display_name" defaultValue={user.display_name} required maxLength={100} /></label><button className="portal-button portal-button--primary" disabled={busy}>Save profile</button></form>
      </section>
      <section className="portal-panel portal-settings"><h2>Change password</h2><p>Use at least 12 characters. Changing your password signs out other sessions.</p>
        <form className="portal-form portal-settings-form" onSubmit={savePassword}><label>Current password<input name="current_password" type="password" autoComplete="current-password" required /></label><label>New password<input name="new_password" type="password" autoComplete="new-password" minLength={12} required /></label><button className="portal-button portal-button--primary" disabled={busy}>Update password</button></form>
      </section>
      <section className="portal-panel portal-settings"><h2>Privacy and clinical safety</h2><p>Account information, uploaded images, and notes are stored on this application server. Image interpretation is performed by the local classifier. Gemini integration is intentionally disabled; no patient data is sent to third-party AI services.</p><p className="portal-disclaimer">This software is an automated research prototype and is not for diagnosis or treatment. Always consult a qualified healthcare professional.</p></section>
    </>
  )
}

export default function App() {
  const [user, setUser] = useState(null)
  const [csrfToken, setCsrfToken] = useState('')
  const [authLoading, setAuthLoading] = useState(true)
  const [authError, setAuthError] = useState('')
  const [backend, setBackend] = useState({ state: 'checking', model: null })
  const [page, setPage] = useState('dashboard')
  const [summary, setSummary] = useState(null)
  const [items, setItems] = useState([])
  const [patients, setPatients] = useState([])
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [historyQuery, setHistoryQuery] = useState('')
  const [historyPrediction, setHistoryPrediction] = useState('')
  const [patientQuery, setPatientQuery] = useState('')
  const [notice, setNotice] = useState('')

  const refreshDashboard = useCallback(async () => {
    setSummary(await fetchDashboard())
  }, [])

  useEffect(() => {
    let active = true
    Promise.all([
      checkHealth().catch((error) => ({ model_ready: false, detail: error.message, offline: true })),
      getCurrentUser().catch((error) => {
        if (error.status !== 401) setAuthError(error.message)
        return null
      }),
    ]).then(([health, session]) => {
      if (!active) return
      setBackend({
        state: health.offline ? 'offline' : health.model_ready ? 'ready' : 'not_trained',
        model: health.model || null,
      })
      if (session) {
        setUser(session.user)
        setCsrfToken(session.csrf_token)
      }
      setAuthLoading(false)
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!user) return
    refreshDashboard().catch((error) => setNotice(error.message))
  }, [user, refreshDashboard])

  useEffect(() => {
    if (!user || page !== 'history') return
    fetchAnalyses({ q: historyQuery, prediction: historyPrediction })
      .then((response) => setItems(response.items))
      .catch((error) => setNotice(error.message))
  }, [user, page, historyQuery, historyPrediction])

  useEffect(() => {
    if (!user || page !== 'patients') return
    fetchPatients(patientQuery).then(setPatients).catch((error) => setNotice(error.message))
  }, [user, page, patientQuery])

  function authenticated(response) {
    setUser(response.user)
    setCsrfToken(response.csrf_token)
    setAuthError('')
    setPage('dashboard')
  }

  async function signOut() {
    try {
      await logoutAccount(csrfToken)
    } catch (error) {
      setNotice(error.message)
      return
    }
    setUser(null); setCsrfToken(''); setSummary(null); setPage('dashboard')
  }

  async function openAnalysis(id) {
    setNotice('')
    setSelectedAnalysis(null)
    setPage('analysis')
    try { setSelectedAnalysis(await fetchAnalysis(id)) } catch (error) { setNotice(error.message) }
  }

  async function openPatient(id) {
    setNotice('')
    setSelectedPatient(null)
    setPage('patient')
    try { setSelectedPatient(await fetchPatient(id)) } catch (error) { setNotice(error.message) }
  }

  async function removeAnalysis(id) {
    if (!window.confirm('Delete this analysis and its stored image?')) return
    try {
      await deleteAnalysis(id, csrfToken)
      setNotice('Analysis deleted.')
      setSelectedAnalysis(null)
      setPage('history')
      await refreshDashboard()
      setItems((current) => current.filter((item) => item.id !== id))
    } catch (error) { setNotice(error.message) }
  }

  async function removePatient(id) {
    if (!window.confirm('Delete this patient record and all linked analyses and images?')) return
    try {
      await deletePatient(id, csrfToken)
      setPatients((current) => current.filter((patient) => patient.id !== id))
      await refreshDashboard()
    } catch (error) { setNotice(error.message) }
  }

  async function savePdf(id) {
    try {
      const blob = await downloadReport(id)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `analysis-${id.slice(0, 8)}.pdf`
      anchor.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) { setNotice(error.message) }
  }

  if (authLoading) return <div className="portal-loading"><span className="portal-loading__spinner" aria-hidden="true" />Connecting to your private workspace…</div>
  if (!user) return <><AuthScreen onAuthenticated={authenticated} /><p className="portal-auth-error">{authError}</p></>

  return (
    <div className="portal-app">
      <Topbar user={user} backendState={backend.state} onLogout={signOut} onNavigate={setPage} />
      <div className="portal-shell">
        <Sidebar page={page} onNavigate={(next) => { setNotice(''); setPage(next) }} user={user} />
        <main className="portal-main">
          {notice && <div className="portal-alert portal-alert--info" role="status"><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss">×</button></div>}
          {page === 'dashboard' && <Overview summary={summary} backend={backend} user={user} onNavigate={setPage} onOpenAnalysis={openAnalysis} />}
          {page === 'new' && <NewAnalysis csrfToken={csrfToken} onCreated={async (record) => { await refreshDashboard(); setSelectedAnalysis(record); setPage('analysis') }} />}
          {page === 'history' && <History items={items} query={historyQuery} setQuery={setHistoryQuery} prediction={historyPrediction} setPrediction={setHistoryPrediction} onOpen={openAnalysis} />}
          {page === 'patients' && <Patients patients={patients} query={patientQuery} setQuery={setPatientQuery} onOpen={openPatient} onDelete={removePatient} />}
          {page === 'analysis' && <AnalysisDetail analysis={selectedAnalysis} onBack={() => setPage('history')} onNew={() => setPage('new')} onDelete={removeAnalysis} onDownload={savePdf} />}
          {page === 'patient' && <PatientDetail patient={selectedPatient} onBack={() => setPage('patients')} onOpenAnalysis={openAnalysis} />}
          {page === 'settings' && <Settings user={user} csrfToken={csrfToken} onUserUpdated={setUser} onCsrfUpdated={setCsrfToken} />}
          <footer className="portal-footer">CareView Imaging · Research prototype · Not for clinical diagnosis</footer>
        </main>
      </div>
    </div>
  )
}
