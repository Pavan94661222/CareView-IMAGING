import { useState } from 'react'
import { confirmPasswordReset, loginAccount, registerAccount, requestPasswordReset } from '../api.js'

export default function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [resetSent, setResetSent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  async function submit(event) {
    event.preventDefault()
    setBusy(true)
    setError('')
    setMessage('')
    const values = Object.fromEntries(new FormData(event.currentTarget).entries())
    try {
      if (mode === 'login') {
        onAuthenticated(await loginAccount(values))
      } else if (mode === 'register') {
        onAuthenticated(await registerAccount(values))
      } else if (!resetSent) {
        const result = await requestPasswordReset(values.email)
        setResetSent(true)
        setMessage(result.message)
      } else {
        await confirmPasswordReset(values.token, values.new_password)
        setMode('login')
        setResetSent(false)
        setMessage('Password updated. Sign in with your new password.')
      }
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  const isForgot = mode === 'forgot'
  const showEmail = mode === 'login' || mode === 'register' || (isForgot && !resetSent)
  const title = mode === 'login' ? 'Welcome back' : mode === 'register' ? 'Create your account' : 'Reset password'

  return (
    <main className="portal-auth">
      <section className="portal-auth__card">
        <div className="portal-brand portal-brand--auth">
          <span className="portal-brand__mark" aria-hidden="true">+</span>
          <span>CareView <small>IMAGING</small></span>
        </div>
        <p className="portal-eyebrow">Private clinical workspace</p>
        <h1>{title}</h1>
        <p className="portal-muted">
          {isForgot
            ? 'Request a one-time reset code by email.'
            : mode === 'register'
              ? 'Create an account to securely manage your local records.'
              : 'Sign in to access your patient records and image analyses.'}
        </p>

        <form className="portal-form" onSubmit={submit}>
          {showEmail && (
            <label>
              Email address
              <input name="email" type="email" autoComplete="email" required maxLength={254} />
            </label>
          )}
          {mode === 'register' && (
            <label>
              Full name
              <input name="display_name" autoComplete="name" required maxLength={100} />
            </label>
          )}
          {mode === 'login' && (
            <label>
              Password
              <input name="password" type="password" autoComplete="current-password" required />
            </label>
          )}
          {mode === 'register' && (
            <label>
              Password <span className="portal-field-hint">(at least 12 characters)</span>
              <input name="password" type="password" autoComplete="new-password" minLength={12} required />
            </label>
          )}
          {isForgot && resetSent && (
            <>
              <label>
                Reset code
                <input name="token" autoComplete="one-time-code" required minLength={20} />
              </label>
              <label>
                New password <span className="portal-field-hint">(at least 12 characters)</span>
                <input name="new_password" type="password" autoComplete="new-password" minLength={12} required />
              </label>
            </>
          )}

          {error && <p className="portal-alert portal-alert--error" role="alert">{error}</p>}
          {message && <p className="portal-alert portal-alert--success" role="status">{message}</p>}
          <button className="portal-button portal-button--primary portal-button--wide" disabled={busy}>
            {busy ? 'Please wait…' : isForgot ? (resetSent ? 'Update password' : 'Send reset code') : mode === 'register' ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <div className="portal-auth__links">
          {mode === 'login' && (
            <>
              <button type="button" className="portal-text-button" onClick={() => { setMode('forgot'); setError(''); setMessage('') }}>
                Forgot password?
              </button>
              <p>New to CareView? <button type="button" className="portal-text-button" onClick={() => { setMode('register'); setError(''); setMessage('') }}>Create an account</button></p>
            </>
          )}
          {mode === 'register' && (
            <p>Already registered? <button type="button" className="portal-text-button" onClick={() => { setMode('login'); setError(''); setMessage('') }}>Sign in</button></p>
          )}
          {isForgot && (
            <button type="button" className="portal-text-button" onClick={() => { setMode('login'); setResetSent(false); setError(''); setMessage('') }}>
              Back to sign in
            </button>
          )}
        </div>
        <p className="portal-auth__privacy">Your account and health records are stored on this server. Patient data is not sent to Gemini or another external AI service.</p>
      </section>
    </main>
  )
}
