import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Chat from './components/Chat.jsx'
import Input from './components/Input.jsx'
import Citations from './components/Citations.jsx'
import { streamInvestigate, fetchCompanies } from './api.js'

let msgId = 0
const uid = () => ++msgId

export default function App() {
  const [companies, setCompanies]   = useState([])
  const [messages, setMessages]     = useState([])
  const [streaming, setStreaming]   = useState(false)
  const [citations, setCitations]   = useState([])
  const [graphData, setGraphData]   = useState(null)
  const [evalScores, setEvalScores] = useState(null)

  useEffect(() => {
    fetchCompanies().then(setCompanies).catch(() => setCompanies([]))
  }, [])

  const pushMsg = (msg) => setMessages(prev => [...prev, msg])

  const send = useCallback(async (query) => {
    if (streaming) return

    setCitations([])
    setGraphData(null)
    setEvalScores(null)
    setStreaming(true)

    pushMsg({ id: uid(), type: 'user', content: query })

    let stageId = uid()
    let activeRoute = null

    pushMsg({ id: stageId, type: 'stage', stage: 'planning' })

    await streamInvestigate(query, {
      onStage: (stage, route) => {
        if (route) activeRoute = route
        setMessages(prev => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.type === 'stage') {
            next[next.length - 1] = { id: stageId, type: 'stage', stage, route: activeRoute }
          }
          return next
        })
      },

      onComplete: (payload) => {
        setMessages(prev => prev.filter(m => m.id !== stageId))
        pushMsg({
          id: uid(),
          type: 'assistant',
          content: payload.answer,
          route: activeRoute,
          streaming: false,
        })
        if (payload.citations?.length)  setCitations(payload.citations)
        if (payload.graph_data?.nodes?.length) setGraphData(payload.graph_data)
        setEvalScores({ route: activeRoute })
        setStreaming(false)
      },

      onError: (detail) => {
        setMessages(prev => prev.filter(m => m.id !== stageId))
        pushMsg({ id: uid(), type: 'error', content: detail })
        setStreaming(false)
      },
    })
  }, [streaming])

  return (
    <div className="layout">
      <Sidebar companies={companies} onQuery={send} />
      <main className="chat-main">
        <div className="chat-topbar">
          <span className="topbar-title">Investigation workspace</span>
          <span className="topbar-badge">LangGraph · Neo4j · Qdrant</span>
        </div>
        <Chat messages={messages} />
        <Input onSend={send} disabled={streaming} />
      </main>
      <Citations citations={citations} evalScores={evalScores} graphData={graphData} />
    </div>
  )
}