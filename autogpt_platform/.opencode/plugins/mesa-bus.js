// Mesa context-bus plugin (auto-generated — do not edit).
export const MesaBus = async ({ client, directory }) => {
  const PORT = process.env.MESA_BUS_PORT
  const CANVAS = process.env.MESA_BUS_CANVAS
  const NODE = process.env.MESA_BUS_NODE
  if (!PORT || !CANVAS || !NODE) return {}
  const base = 'http://127.0.0.1:' + PORT

  const post = async (path, body) => {
    try {
      await fetch(base + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    } catch { }
  }
  const textOf = (parts) =>
    Array.isArray(parts)
      ? parts.filter((p) => p && p.type === 'text' && typeof p.text === 'string').map((p) => p.text).join('\n')
      : ''

  void post('/hook/session', { canvas: CANVAS, node: NODE, status: 'live', agent: 'opencode' })

  return {
    // before each user prompt: pull queued Mesa messages and ride along as an extra part
    'chat.message': async (_input, output) => {
      try {
        const r = await fetch(base + '/hook/pre-prompt?canvas=' + encodeURIComponent(CANVAS) + '&node=' + encodeURIComponent(NODE))
        const text = await r.text()
        if (text && text.trim() && output && Array.isArray(output.parts)) output.parts.push({ type: 'text', text })
      } catch { }
    },

    'permission.ask': async (input) => {
      const message = input && typeof input.title === 'string' && input.title ? input.title : 'Wants to run a tool'
      await post('/hook/notify', { canvas: CANVAS, node: NODE, message })
    },

    event: async ({ event }) => {
      try {
        if (!event || event.type !== 'session.idle') return
        const id = event.properties && event.properties.sessionID
        if (!id) return
        try {
          const info = await client.session.get({ path: { id } })
          if (info && info.data && info.data.parentID) return // subagent sessions are internal
        } catch { }
        const res = await client.session.messages({ path: { id } })
        const msgs = (res && res.data) || []
        let userPrompt = ''
        let assistantText = ''
        for (const m of msgs) {
          const role = m && m.info && m.info.role
          if (role === 'user') {
            const t = textOf(m.parts)
            if (t.trim()) { userPrompt = t; assistantText = '' }
          } else if (role === 'assistant') {
            const t = textOf(m.parts)
            if (t.trim()) assistantText += (assistantText ? '\n' : '') + t
          }
        }
        if (!assistantText.trim()) return
        await post('/hook/stop', {
          canvas: CANVAS,
          node: NODE,
          agent: 'opencode',
          excerpt: { cwd: directory, userPrompt: userPrompt.slice(0, 6000), assistantText: assistantText.slice(0, 12000) },
        })
      } catch { }
    },
  }
}
