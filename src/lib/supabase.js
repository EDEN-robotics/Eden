import { createClient } from '@supabase/supabase-js'

// Trim whitespace so a trailing newline in a pasted env var doesn't break
// the realtime WebSocket (the key ends up URL-encoded as %0A and the server
// rejects the handshake).
const supabaseUrl = (import.meta.env.VITE_SUPABASE_URL || '').trim()
const supabaseAnonKey = (import.meta.env.VITE_SUPABASE_ANON_KEY || '').trim()

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
