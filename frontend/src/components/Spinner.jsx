export default function Spinner({ label }) {
  return (
    <div className="spinner-wrap" role="status">
      <span className="spinner" aria-hidden="true" />
      {label && <span className="spinner__label">{label}</span>}
    </div>
  )
}
