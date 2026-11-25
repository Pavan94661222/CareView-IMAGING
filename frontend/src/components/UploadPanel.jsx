import { useRef, useState } from 'react'

const ACCEPT_ATTR = 'image/jpeg,image/png,image/bmp,image/tiff'

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${index === 0 || value >= 10 ? Math.round(value) : value.toFixed(1)} ${units[index]}`
}

function CloudIcon() {
  return (
    <svg viewBox="0 0 48 48" width="46" height="46" aria-hidden="true" focusable="false">
      <g fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M32.5 30.5a6.5 6.5 0 0 0 0-13 8 8 0 0 0-15.4-2.4A6 6 0 0 0 12.5 30.5" />
        <path d="M24 22v15" />
        <path d="M18.5 27.5 24 22l5.5 5.5" />
      </g>
    </svg>
  )
}

export default function UploadPanel({
  file,
  previewUrl,
  dimensions,
  validationError,
  loading,
  canDetect,
  backendState,
  onFileSelected,
  onClear,
  onDetect,
}) {
  const [isDragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const inputRef = useRef(null)

  const openPicker = () => inputRef.current?.click()

  const handleInputChange = (event) => {
    const selected = event.target.files && event.target.files[0]
    if (selected) onFileSelected(selected)
    event.target.value = ''
  }

  const handleDragEnter = (event) => {
    event.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) {
      dragDepth.current = 0
      setDragging(false)
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const dropped = event.dataTransfer?.files?.[0]
    if (dropped) onFileSelected(dropped)
  }

  const disabledHint = (() => {
    if (loading) return 'Analysing the current image…'
    if (!file) return 'Select a chest X-ray image first.'
    if (backendState === 'checking') return 'Checking backend status…'
    if (backendState === 'not_trained') {
      return 'The model is not trained. Run "python -m pneumonia.train" on the backend.'
    }
    if (backendState === 'offline') {
      return 'The backend is offline. Start the API on http://127.0.0.1:8000.'
    }
    return ''
  })()

  return (
    <div className="card upload-card">
      <div className="card__header">
        <span className="eyebrow">Upload image</span>
        {file && (
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={onClear}
            disabled={loading}
          >
            Clear
          </button>
        )}
      </div>

      {!file ? (
        <div
          className={`dropzone${isDragging ? ' dropzone--active' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={openPicker}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              openPicker()
            }
          }}
          role="button"
          tabIndex={0}
          aria-label="Upload a chest X-ray: drag a file here or press Enter to browse"
        >
          <span className="dropzone__icon">
            <CloudIcon />
          </span>
          <p className="dropzone__title">Drag and drop a chest X-ray here</p>
          <p className="dropzone__or">or</p>
          <span className="button button--primary">Choose image</span>
          <p className="dropzone__meta">JPEG, PNG, BMP or TIFF — up to 10 MB</p>
        </div>
      ) : (
        <div className="preview">
          <div className="preview__frame">
            <img className="preview__image" src={previewUrl} alt={`Preview of the selected chest X-ray: ${file.name}`} />
          </div>
          <div className="preview__meta">
            <p className="preview__name" title={file.name}>
              {file.name}
            </p>
            <p className="preview__details">
              {formatBytes(file.size)}
              {dimensions ? ` • ${dimensions.width} × ${dimensions.height} px` : ''}
            </p>
          </div>
          <button
            type="button"
            className="button button--ghost button--small"
            onClick={openPicker}
            disabled={loading}
          >
            Choose a different image
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="visually-hidden"
        onChange={handleInputChange}
        tabIndex={-1}
      />

      {validationError && (
        <p className="field-error" role="alert">
          {validationError}
        </p>
      )}

      <button
        type="button"
        className="button button--primary button--block detect-button"
        onClick={onDetect}
        disabled={!canDetect}
        title={disabledHint || 'Run the model on the selected image'}
      >
        Detect Pneumonia
      </button>

      {!canDetect && disabledHint && <p className="detect-hint">{disabledHint}</p>}
    </div>
  )
}
