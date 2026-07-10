import { useEffect, useRef, useState } from 'react'

const TYPE_COLOR = {
  Company:    '#1a9e75',
  Auditor:    '#7f77dd',
  Subsidiary: '#ba7517',
  RiskFactor: '#e24b4a',
  Person:     '#378add',
}
const TYPE_RADIUS = {
  Company: 22, Auditor: 18, Subsidiary: 16, RiskFactor: 16, Person: 14,
}

export default function GraphViz({ graphData }) {
  const containerRef = useRef(null)
  const svgRef       = useRef(null)
  const simRef       = useRef(null)
  const [selected, setSelected] = useState(null)
  const [d3Ready, setD3Ready]   = useState(!!window.d3)

  // Load D3 once from CDN
  useEffect(() => {
    if (window.d3) { setD3Ready(true); return }
    const s = document.createElement('script')
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js'
    s.onload = () => setD3Ready(true)
    document.head.appendChild(s)
  }, [])

  useEffect(() => {
    if (!d3Ready || !graphData?.nodes?.length || !svgRef.current || !containerRef.current) return

    const d3  = window.d3
    const W   = containerRef.current.clientWidth  || 640
    const H   = containerRef.current.clientHeight || 400

    // Stop previous simulation
    if (simRef.current) simRef.current.stop()
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', W).attr('height', H)

    // Arrowhead markers
    const defs = svg.append('defs')
    Object.entries({
      AUDITED_BY: '#7f77dd', HAS_SUBSIDIARY: '#ba7517',
      HAS_RISK: '#e24b4a', default: '#444'
    }).forEach(([key, color]) => {
      defs.append('marker')
        .attr('id', `arr-${key}`).attr('viewBox', '0 -4 8 8')
        .attr('refX', 30).attr('refY', 0)
        .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
        .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', color)
    })

    const nodes = graphData.nodes.map(n => ({ ...n }))
    const links = graphData.edges.map(e => ({ ...e }))

    const sim = d3.forceSimulation(nodes)
      .force('link',    d3.forceLink(links).id(d => d.id).distance(120).strength(0.9))
      .force('charge',  d3.forceManyBody().strength(-400))
      .force('center',  d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(40))
    simRef.current = sim

    const g = svg.append('g')
    svg.call(d3.zoom().scaleExtent([0.3, 3])
      .on('zoom', e => g.attr('transform', e.transform)))

    // Edges
    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => ({ AUDITED_BY: '#7f77dd', HAS_SUBSIDIARY: '#ba7517', HAS_RISK: '#e24b4a' }[d.label] || '#444')  )
      .attr('stroke-width', 1.5).attr('stroke-opacity', 0.8)
      .attr('marker-end', d => `url(#arr-${d.label || 'default'})`)

    // Edge labels
    const edgeLabel = g.append('g').selectAll('text').data(links).join('text')
      .attr('font-size', 9).attr('fill', '#555')
      .attr('text-anchor', 'middle').attr('dy', -5)
      .style('pointer-events', 'none').text(d => d.label)

    // Node groups
    const node = g.append('g').selectAll('g').data(nodes).join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y })
        .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )
      .on('click', (e, d) => { e.stopPropagation(); setSelected(d) })

    node.append('circle')
      .attr('r', d => TYPE_RADIUS[d.type] || 16)
      .attr('fill', d => (TYPE_COLOR[d.type] || '#555') + '25')
      .attr('stroke', d => TYPE_COLOR[d.type] || '#555')
      .attr('stroke-width', 2)

    // Type initial inside circle
    node.append('text')
      .attr('text-anchor', 'middle').attr('dy', '0.35em')
      .attr('font-size', d => (TYPE_RADIUS[d.type] || 16) * 0.65)
      .attr('fill', d => TYPE_COLOR[d.type] || '#ccc')
      .attr('font-weight', '700').attr('font-family', 'Inter, sans-serif')
      .style('pointer-events', 'none')
      .text(d => d.type?.[0] || '?')

    // Node name below circle
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', d => (TYPE_RADIUS[d.type] || 16) + 13)
      .attr('font-size', 10).attr('fill', '#bbb').attr('font-family', 'Inter, sans-serif')
      .style('pointer-events', 'none')
      .each(function(d) {
        const el = d3.select(this)
        const words = d.label.split(' ')
        if (words.length <= 2) {
          el.text(d.label)
        } else {
          el.append('tspan').attr('x', 0).attr('dy', 0).text(words.slice(0, 2).join(' '))
          el.append('tspan').attr('x', 0).attr('dy', 12).text(words.slice(2).join(' '))
        }
      })

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
      edgeLabel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2)
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    svg.on('click', () => setSelected(null))
    return () => sim.stop()
  }, [d3Ready, graphData])

  if (!graphData?.nodes?.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Legend */}
      <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 12, flexWrap: 'wrap', flexShrink: 0 }}>
        {Object.entries(TYPE_COLOR).map(([t, c]) => (
          <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: c }} />
            <span style={{ fontSize: 10, color: 'var(--text-3)' }}>{t}</span>
          </div>
        ))}
      </div>

      {/* Canvas — takes all remaining space */}
      <div ref={containerRef} style={{ flex: 1, position: 'relative', minHeight: 0 }}>
        <svg ref={svgRef} style={{ width: '100%', height: '100%', display: 'block' }} />

        {/* Click-selected node detail */}
        {selected && (
          <div style={{
            position: 'absolute', bottom: 8, left: 8, right: 8,
            background: 'var(--bg-2)', border: `1px solid ${TYPE_COLOR[selected.type] || 'var(--border)'}`,
            borderRadius: 'var(--radius)', padding: '10px 12px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{
                fontSize: 10, padding: '1px 6px', borderRadius: 4, fontWeight: 600,
                background: (TYPE_COLOR[selected.type] || '#555') + '22',
                color: TYPE_COLOR[selected.type] || '#ccc',
              }}>{selected.type}</span>
              <button onClick={() => setSelected(null)} style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: 16 }}>×</button>
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>{selected.label}</div>
            {selected.note     && <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 2 }}>{selected.note}</div>}
            {selected.years    && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Years active: {selected.years}</div>}
            {selected.subtype  && <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Type: {selected.subtype}</div>}
            {selected.severity && (
              <div style={{ fontSize: 11, marginTop: 2 }}>
                Severity: <span style={{ color: selected.severity === 'critical' ? 'var(--danger)' : 'var(--warning)', fontWeight: 600 }}>{selected.severity}</span>
              </div>
            )}
          </div>
        )}
      </div>

      <div style={{ padding: '5px 12px', borderTop: '1px solid var(--border)', fontSize: 10, color: 'var(--text-3)', flexShrink: 0 }}>
        {graphData.nodes.length} nodes · {graphData.edges.length} edges · drag · scroll to zoom · click node for details
      </div>
    </div>
  )
}