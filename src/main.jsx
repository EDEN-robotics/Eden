import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.jsx'
import Documentation from './pages/Documentation.jsx'
import Chat from './pages/Chat.jsx'
import Simulator from './pages/Simulator.jsx'
import Architecture from './pages/Architecture.jsx'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY
const hasClerkKey = PUBLISHABLE_KEY && !PUBLISHABLE_KEY.includes('YOUR_KEY_HERE')

function AppContent() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/docs/*" element={<Documentation />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/sim" element={<Simulator />} />
        <Route path="/architecture" element={<Architecture />} />
      </Routes>
    </BrowserRouter>
  )
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {hasClerkKey ? (
      <ClerkProvider publishableKey={PUBLISHABLE_KEY} signInForceRedirectUrl="https://eden-robotics.github.io/Eden/chat" signUpForceRedirectUrl="https://eden-robotics.github.io/Eden/chat" signOutForceRedirectUrl="https://eden-robotics.github.io/Eden/">
        <AppContent />
      </ClerkProvider>
    ) : (
      <AppContent />
    )}
  </StrictMode>,
)
