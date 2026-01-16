import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import mkcert from 'vite-plugin-mkcert'

// https://vite.dev/config/
export default defineConfig({
  base: '/Eden/',  // GitHub Pages base path
  server: {
    https: true
  },
  plugins: [
    react(),
    tailwindcss(),
    mkcert(),
  ],
})
