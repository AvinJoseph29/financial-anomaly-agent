import { useEffect, useRef } from 'react'
import Message from './Message.jsx'

export default function Chat({ messages }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="chat-thread">
        <div className="empty-state">
          <div className="empty-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent)' }}>
              <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
            </svg>
          </div>
          <div className="empty-title">Financial Anomaly Agent</div>
          <div className="empty-sub">
            Ask about Enron's SPE structures, SVB's interest rate risk, or Apple's financials.
            Backed by real SEC 10-K filings.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-thread">
      {messages.map(msg => (
        <Message key={msg.id} msg={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
