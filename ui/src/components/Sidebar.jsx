const QUICK_QUERIES = [
  { label: 'Enron Z-score 2000', query: 'What is the Altman Z-score for Enron in 2000?' },
  { label: "SVB's auditor", query: 'Who audited SVB Financial Group?' },
  { label: "Enron's subsidiaries", query: 'What subsidiaries did Enron have?' },
  { label: 'SVB interest rate risk', query: 'What risk factors does SVB disclose about interest rates?' },
  { label: 'Apple supply chain', query: "What does Apple's 10-K say about supply chain risk?" },
  { label: 'Enron related parties', query: "What does Enron's 10-K say about related party transactions?" },
  { label: 'SVB current ratio', query: "What is SVB's current ratio for 2022?" },
  { label: 'Apple debt to equity', query: 'Calculate the debt to equity ratio for Apple in 2023.' },
  { label: 'Who audited Enron?', query: 'Who audited Enron and what happened to them?' },
]

const STATUS_MAP = {
  active:       { label: 'Active',    cls: 'status-active' },
  failed_2023:  { label: 'Failed \'23', cls: 'status-failed' },
  bankrupt_2001:{ label: 'Bankrupt \'01', cls: 'status-bankrupt' },
}

export default function Sidebar({ companies, onQuery }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-dot" />
          <span className="sidebar-title">Anomaly Agent</span>
        </div>
        <div className="sidebar-sub">SEC 10-K Investigation</div>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Companies</div>
        {companies.map(co => {
          const s = STATUS_MAP[co.status] || { label: co.status, cls: 'status-active' }
          return (
            <div key={co.key} className="company-card" onClick={() => onQuery(`Tell me about ${co.name}`)}>
              <div className="company-card-top">
                <span className="company-name">{co.name}</span>
                <span className={`company-status ${s.cls}`}>{s.label}</span>
              </div>
              <span className="company-ticker">{co.ticker} · {co.sector}</span>
            </div>
          )
        })}
      </div>

      <div className="sidebar-section" style={{ paddingBottom: 4 }}>
        <div className="sidebar-label">Quick queries</div>
      </div>
      <div className="sidebar-queries">
        {QUICK_QUERIES.map(q => (
          <button key={q.query} className="query-chip" onClick={() => onQuery(q.query)}>
            {q.label}
          </button>
        ))}
      </div>
    </aside>
  )
}
