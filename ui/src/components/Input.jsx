import { useRef, useEffect } from 'react'

export default function Input({ onSend, disabled }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!disabled) ref.current?.focus()
  }, [disabled])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const submit = () => {
    const val = ref.current?.value.trim()
    if (!val || disabled) return
    ref.current.value = ''
    resize()
    onSend(val)
  }

  const resize = () => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  return (
    <div className="input-area">
      <div className="input-row">
        <textarea
          ref={ref}
          className="input-field"
          placeholder="Ask about Enron's Z-score, SVB's auditor, Apple's supply chain risk..."
          rows={1}
          onKeyDown={handleKey}
          onInput={resize}
          disabled={disabled}
        />
        <button className="send-btn" onClick={submit} disabled={disabled} aria-label="Send">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
      <div className="input-hint">⏎ to send · Shift+⏎ for new line</div>
    </div>
  )
}
