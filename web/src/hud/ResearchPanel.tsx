/**
 * The agent research panel.
 *
 * Two rules govern what this shows.
 *
 * **Trading data keeps priority.** Research is a slow side channel. The panel is
 * closed by default, opens over the world rather than displacing any trading
 * readout, and renders nothing at all when the research pipeline has never run
 * — an empty research panel on a fresh install is noise, not information.
 *
 * **Synthetic results are labelled as synthetic, prominently.** A display
 * surface is exactly where a simulator number gets mistaken for evidence about
 * a live market, so the provenance banner is the first thing in the panel and
 * it is not dismissible.
 */

import { useEffect } from 'react'
import { useShallow } from 'zustand/react/shallow'

import { api } from '../net/rest'
import type { AgentResearch } from '../net/protocol'
import { useWorldStore } from '../state/store'

const BUSY_STATES = new Set(['PROPOSED', 'TRAINING', 'VALIDATING'])

function statusLabel(agent: AgentResearch): string {
  if (!agent.current_status) return 'no experiments yet'
  if (agent.current_status === 'REJECTED') return 'last hypothesis rejected'
  if (agent.current_status === 'CANDIDATE') return 'candidate awaiting review'
  if (agent.current_status === 'PROMOTED') return 'champion'
  return agent.current_status.toLowerCase()
}

function money(value: number): string {
  return value >= 1000 ? `$${(value / 1000).toFixed(1)}k` : `$${value.toFixed(0)}`
}

export function ResearchPanel(): JSX.Element | null {
  const { research, showResearch, setResearch, toggleResearch } = useWorldStore(
    useShallow((s) => ({
      research: s.research,
      showResearch: s.showResearch,
      setResearch: s.setResearch,
      toggleResearch: s.toggleResearch,
    })),
  )

  useEffect(() => {
    if (!showResearch || research) return
    let cancelled = false
    api
      .research()
      .then((state) => {
        if (!cancelled) setResearch(state)
      })
      .catch(() => {
        /* Research is optional. A missing endpoint must not break the world. */
      })
    return () => {
      cancelled = true
    }
  }, [showResearch, research, setResearch])

  if (!showResearch) return null

  if (!research || !research.available) {
    return (
      <div className="research-panel" role="dialog" aria-label="Agent research">
        <header>
          <h2>AGENT RESEARCH</h2>
          <button type="button" onClick={toggleResearch} aria-label="Close research">
            ×
          </button>
        </header>
        <p className="research-empty">
          No research has run yet. Start one with{' '}
          <code>python research.py campaign --agent KIRA</code>.
        </p>
      </div>
    )
  }

  return (
    <div className="research-panel" role="dialog" aria-label="Agent research">
      <header>
        <h2>AGENT RESEARCH</h2>
        <button type="button" onClick={toggleResearch} aria-label="Close research">
          ×
        </button>
      </header>

      <p className={research.synthetic_only ? 'research-banner synthetic' : 'research-banner real'}>
        {research.synthetic_only
          ? 'SYNTHETIC — simulator output. Not evidence about live markets.'
          : `REAL MARKET DATA — ${research.sources.filter((s) => s !== 'SYNTHETIC').join(', ')}`}
      </p>

      <dl className="research-totals">
        <div>
          <dt>datasets</dt>
          <dd>{research.datasets}</dd>
        </div>
        <div>
          <dt>experiments</dt>
          <dd>{research.experiments}</dd>
        </div>
        <div>
          <dt>policies</dt>
          <dd>{research.policies}</dd>
        </div>
        <div>
          <dt>paper deployed</dt>
          <dd>{money(research.paper_capital_deployed)}</dd>
        </div>
      </dl>

      <table className="research-table">
        <thead>
          <tr>
            <th>agent</th>
            <th>champion</th>
            <th className="num">chal.</th>
            <th className="num">exp.</th>
            <th className="num">paper</th>
            <th>state</th>
          </tr>
        </thead>
        <tbody>
          {research.agents.map((agent) => (
            <tr
              key={agent.agent}
              className={BUSY_STATES.has(agent.current_status ?? '') ? 'busy' : undefined}
            >
              <td>{agent.agent}</td>
              <td className={agent.champion_policy ? 'champion' : 'none'}>
                {agent.champion_policy ? `v${agent.champion_version ?? '?'}` : '—'}
              </td>
              <td className="num">{agent.challenger_count}</td>
              <td className="num">
                {agent.experiments_total}
                {agent.experiments_rejected > 0 && (
                  <span className="rejected"> ({agent.experiments_rejected} rej)</span>
                )}
              </td>
              <td className="num">{money(agent.paper_allocation)}</td>
              <td className="state">{statusLabel(agent)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="research-footnote">
        Promotion here is a research status. Nothing reaches live trading without the
        three-part gate and explicit approval, per policy and per venue.
      </p>
    </div>
  )
}
