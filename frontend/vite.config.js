import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend talks to the FastAPI backend through the dev-server proxy so that
// the app can simply call relative paths such as `/api/predict`.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
