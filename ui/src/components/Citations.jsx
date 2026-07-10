import GraphViz from './GraphViz.jsx'

export default function Citations({ citations, evalScores, graphData }) {
  const hasGraph     = graphData?.nodes?.length > 0
  const hasCitations = citations?.length > 0

  return (
    <div className="citations-panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', height: '100%' }}>

      <div className="citations-header" style={{ flexShrink: 0 }}>
        {hasGraph ? 'Knowledge Graph Evidence' : 'Sources & Eval'}
      </div>

      {/* Graph view */}
      {hasGraph && (
        <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <GraphViz graphData={graphData} />
        </div>
      )}

      {/* Citations view */}
      {!hasGraph && (
        <div className="citations-body" style={{ flex: 1, overflowY: 'auto' }}>
          {!hasCitations ? (
            <div className="empty-citations">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-3)' }}>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
              <span className="empty-citations-text">
                Citations appear here after retrieve queries. The knowledge graph appears here after graph queries.
              </span>
            </div>
          ) : (
            citations.map((c, i) => (
              <div key={i} className="citation-card">
                <div className="citation-company">{c.company}</div>
                <div className="citation-period">10-K · {c.period}</div>
                {c.source_url && (
                  <a href={c.source_url} target="_blank" rel="noreferrer" className="citation-link">
                    SEC EDGAR ↗
                  </a>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Route badge — always at bottom */}
      {evalScores?.route && (
        <div className="eval-panel" style={{ flexShrink: 0 }}>
          <div className="eval-label">Route info</div>
          <div className="eval-row">
            <span className="eval-name">Route taken</span>
            <span className={`message-route route-${evalScores.route}`} style={{ fontSize: 10 }}>
              {evalScores.route}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}