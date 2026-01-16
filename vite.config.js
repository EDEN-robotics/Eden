import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(async ({ mode }) => {
  const plugins = [react(), tailwindcss()]

  // Only load mkcert in development (for local HTTPS)
  if (mode === 'development') {
    const mkcert = (await import('vite-plugin-mkcert')).default
    plugins.push(mkcert())
  }

  return {
    base: '/Eden/',  // GitHub Pages base path
    server: {
      https: true
    },
    plugins,
  }
})
