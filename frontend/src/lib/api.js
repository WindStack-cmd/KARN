import axios from 'axios'

// Accept either a bare backend origin (for example http://127.0.0.1:8000)
// or an already-prefixed API URL. All KARN data routes live below /api.
const configuredUrl = import.meta.env.VITE_API_URL?.replace(/\/$/, '')
const baseURL = configuredUrl
  ? (configuredUrl.endsWith('/api') ? configuredUrl : `${configuredUrl}/api`)
  : 'http://127.0.0.1:8000/api'

export const api = axios.create({ baseURL })
