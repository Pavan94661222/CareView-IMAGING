const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const OFFLINE_MESSAGE =
  'Could not reach the backend. Is the API running on http://127.0.0.1:8000?'

export class ApiError extends Error {
  constructor(message, { code = null, detail = null, status = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.detail = detail
    this.status = status
  }
}

function responseErrorMessage(payload, fallback) {
  if (typeof payload?.error === 'string') return payload.error
  if (typeof payload?.detail === 'string') return payload.detail
  if (Array.isArray(payload?.detail)) return 'Please check the submitted information and try again.'
  return fallback
}

async function request(path, { method = 'GET', body, csrfToken, headers = {} } = {}) {
  const requestHeaders = { Accept: 'application/json', ...headers }
  if (csrfToken) requestHeaders['X-CSRF-Token'] = csrfToken
  if (body && !(body instanceof FormData)) requestHeaders['Content-Type'] = 'application/json'

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      body: body && !(body instanceof FormData) ? JSON.stringify(body) : body,
      headers: requestHeaders,
      credentials: 'include',
    })
  } catch {
    throw new ApiError(OFFLINE_MESSAGE)
  }

  if (response.status === 204) return null
  let payload = null
  try {
    payload = await response.json()
  } catch {
    if (response.ok) throw new ApiError('The server returned a malformed response.')
  }
  if (!response.ok) {
    const message = responseErrorMessage(
      payload,
      `The server returned an unexpected response (status ${response.status}).`,
    )
    throw new ApiError(message, {
      code: payload?.code ?? null,
      detail: payload?.detail ?? null,
      status: response.status,
    })
  }
  return payload
}

export function checkHealth() {
  return request('/api/health')
}

export function fetchMetrics() {
  return request('/api/metrics')
}

export function predict(file) {
  const body = new FormData()
  body.append('file', file)
  return request('/api/predict', { method: 'POST', body })
}

export function registerAccount(payload) {
  return request('/api/auth/register', { method: 'POST', body: payload })
}

export function loginAccount(payload) {
  return request('/api/auth/login', { method: 'POST', body: payload })
}

export function getCurrentUser() {
  return request('/api/auth/me')
}

export function logoutAccount(csrfToken) {
  return request('/api/auth/logout', { method: 'POST', csrfToken })
}

export function requestPasswordReset(email) {
  return request('/api/auth/password-reset/request', {
    method: 'POST',
    body: { email },
  })
}

export function confirmPasswordReset(token, new_password) {
  return request('/api/auth/password-reset/confirm', {
    method: 'POST',
    body: { token, new_password },
  })
}

export function fetchDashboard() {
  return request('/api/dashboard')
}

export function fetchAnalyses({ q = '', prediction = '', order = 'newest', limit = 100 } = {}) {
  const params = new URLSearchParams({ q, prediction, order, limit: String(limit) })
  return request(`/api/analyses?${params}`)
}

export function createAnalysis(body, csrfToken) {
  return request('/api/analyses', { method: 'POST', body, csrfToken })
}

export function fetchAnalysis(analysisId) {
  return request(`/api/analyses/${encodeURIComponent(analysisId)}`)
}

export function deleteAnalysis(analysisId, csrfToken) {
  return request(`/api/analyses/${encodeURIComponent(analysisId)}`, {
    method: 'DELETE',
    csrfToken,
  })
}

export function fetchPatients(query = '') {
  const params = new URLSearchParams({ q: query })
  return request(`/api/patients?${params}`)
}

export function fetchPatient(patientId) {
  return request(`/api/patients/${encodeURIComponent(patientId)}`)
}

export function deletePatient(patientId, csrfToken) {
  return request(`/api/patients/${encodeURIComponent(patientId)}`, {
    method: 'DELETE',
    csrfToken,
  })
}

export function updateProfile(display_name, csrfToken) {
  return request('/api/auth/profile', {
    method: 'PATCH',
    body: { display_name },
    csrfToken,
  })
}

export function changePassword(current_password, new_password, csrfToken) {
  return request('/api/auth/password', {
    method: 'POST',
    body: { current_password, new_password },
    csrfToken,
  })
}

export async function downloadReport(analysisId) {
  let response
  try {
    response = await fetch(
      `${API_BASE_URL}/api/analyses/${encodeURIComponent(analysisId)}/report.pdf`,
      { credentials: 'include' },
    )
  } catch {
    throw new ApiError(OFFLINE_MESSAGE)
  }
  if (!response.ok) {
    let payload = null
    try {
      payload = await response.json()
    } catch {
      payload = null
    }
    throw new ApiError(responseErrorMessage(payload, 'Could not download the report.'), {
      status: response.status,
    })
  }
  return response.blob()
}
