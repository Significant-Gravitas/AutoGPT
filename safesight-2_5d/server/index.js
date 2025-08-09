const express = require('express')
const path = require('path')

const app = express()
const PORT = process.env.PORT || 8080

const distDir = path.resolve(__dirname, '../dist')
app.use(express.static(distDir, { maxAge: '1d', index: 'index.html' }))

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(distDir, 'index.html'))
})

app.listen(PORT, () => {
  console.log(`[server] static is served from ${distDir}`)
  console.log(`[server] listening on http://localhost:${PORT}`)
})