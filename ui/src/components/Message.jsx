const STAGE_LABELS = {
  planning:     '⟳ Planning route...',
  routed:       '⟳ Route decided',
  gathering:    '⟳ Gathering evidence...',
  synthesising: '⟳ Writing memo...',
}

export default function Message({ msg }) {
  if (msg.type === 'user') {
    return (
      <div className="message">
        <div className="message-meta" style={{ justifyContent: 'flex-end' }}>
          <span className="message-role">You</span>
        </div>
        <div className="message-bubble user">{msg.content}</div>
      </div>
    )
  }

  if (msg.type === 'stage') {
    return (
      <div className="stage-indicator">
        <div className="stage-dot" />
        <span>{STAGE_LABELS[msg.stage] || msg.stage}</span>
        {msg.route && (
          <span className={`message-route route-${msg.route}`}>{msg.route}</span>
        )}
      </div>
    )
  }

  if (msg.type === 'assistant') {
    return (
      <div className="message">
        <div className="message-meta">
          <span className="message-role">Agent</span>
          {msg.route && (
            <span className={`message-route route-${msg.route}`}>{msg.route}</span>
          )}
          {msg.latency && (
            <span style={{ fontSize: 10, color: 'var(--text-3)' }}>{msg.latency}ms</span>
          )}
        </div>
        <div className={`message-bubble ${msg.streaming ? 'streaming' : ''}`}>
          {msg.content}
          {msg.streaming && <span className="cursor" />}
        </div>
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="message">
        <div className="message-meta">
          <span className="message-role" style={{ color: 'var(--danger)' }}>Error</span>
        </div>
        <div className="message-bubble" style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}>
          {msg.content}
        </div>
      </div>
    )
  }

  return null
}
