const BASE = ''

export async function streamInvestigate(query, { onStage, onComplete, onError }) {
  try {
    const res = await fetch(`${BASE}/investigate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    })

    if (!res.ok) throw new Error(`HTTP ${res.status}`)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (line.startsWith('event:')) continue
        if (!line.startsWith('data:')) continue

        const raw = line.slice(5).trim()
        if (!raw || raw === '[DONE]') continue

        try {
          const payload = JSON.parse(raw)
          if (payload.stage === 'complete') {
            onComplete(payload)
          } else if (payload.stage === 'error') {
            onError(payload.detail || 'Unknown error')
          } else {
            onStage(payload.stage, payload.route)
          }
        } catch {
          // skip malformed SSE lines
        }
      }
    }
  } catch (err) {
    onError(err.message)
  }
}

export async function fetchCompanies() {
  const res = await fetch(`${BASE}/companies`)
  return res.json()
}
