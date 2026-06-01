/**
 * WaterfallConfig UI Tests
 *
 * Primary goal: verify that the stale-closure bug (two sequential updateDeal
 * calls each spreading the same stale `wf` snapshot) is fixed. Every linked
 * input pair must issue exactly ONE updateDeal call containing BOTH fields.
 */

import React from 'react'
import { render, screen, fireEvent, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { WaterfallConfigPage } from '../WaterfallConfig'
import { useDealStore } from '../../store/dealStore'

// ---------------------------------------------------------------------------
// Mock the Zustand store so we control state and capture updateDeal calls
// ---------------------------------------------------------------------------

vi.mock('../../store/dealStore')

/**
 * The Input component renders <label> + <input> inside a flex-col container
 * without htmlFor/id association.  Find the input by locating the label text
 * and querying the nearest sibling input.
 */
function getInputByLabel(labelText: string): HTMLInputElement {
  const labels = screen.getAllByText(labelText)
  for (const label of labels) {
    const input = label.parentElement?.querySelector('input')
    if (input) return input as HTMLInputElement
  }
  throw new Error(`No input found for label: "${labelText}"`)
}

function getAllInputsByLabel(labelText: string): HTMLInputElement[] {
  return screen
    .getAllByText(labelText)
    .map((label) => label.parentElement?.querySelector('input'))
    .filter((el): el is HTMLInputElement => el !== null && el !== undefined)
}

const DEFAULT_WF = {
  lp_equity_pct: 90,
  gp_equity_pct: 10,
  preferred_return: 8.0,
  pref_compounding: false,
  gp_catchup: true,
  gp_catchup_rate: 50,
  gp_target_promote_pct: 20,
  tiers: [
    { irr_min: 0, irr_max: 14, lp_split: 80, gp_split: 20 },
    { irr_min: 14, irr_max: 18, lp_split: 70, gp_split: 30 },
    { irr_min: 18, irr_max: 999, lp_split: 60, gp_split: 40 },
  ],
}

const DEFAULT_DEAL = {
  waterfall_config: DEFAULT_WF,
  property_info: { purchase_price: 15_000_000 },
  financing: { ltv_pct: 65 },
}

function setupMock(wf = DEFAULT_WF) {
  const mockUpdateDeal = vi.fn()
  const mockSetStep = vi.fn()
  vi.mocked(useDealStore).mockReturnValue({
    deal: { ...DEFAULT_DEAL, waterfall_config: wf },
    updateDeal: mockUpdateDeal,
    setStep: mockSetStep,
  } as ReturnType<typeof useDealStore>)
  return { mockUpdateDeal, mockSetStep }
}

// ---------------------------------------------------------------------------
// LP / GP Equity split
// ---------------------------------------------------------------------------

describe('LP/GP Equity — linked pair atomicity (bug fix)', () => {
  beforeEach(() => vi.clearAllMocks())

  it('changing LP equity issues exactly ONE updateDeal call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('LP Equity (%)'), { target: { value: '80' } })
    expect(mockUpdateDeal).toHaveBeenCalledTimes(1)
  })

  it('changing GP equity issues exactly ONE updateDeal call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('GP Equity (%)'), { target: { value: '15' } })
    expect(mockUpdateDeal).toHaveBeenCalledTimes(1)
  })

  it('LP=80 → both lp_equity_pct=80 and gp_equity_pct=20 in the same call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('LP Equity (%)'), { target: { value: '80' } })
    const wf = mockUpdateDeal.mock.calls[0][0].waterfall_config
    expect(wf.lp_equity_pct).toBe(80)
    expect(wf.gp_equity_pct).toBe(20)
  })

  it('GP=15 → both gp_equity_pct=15 and lp_equity_pct=85 in the same call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('GP Equity (%)'), { target: { value: '15' } })
    const wf = mockUpdateDeal.mock.calls[0][0].waterfall_config
    expect(wf.gp_equity_pct).toBe(15)
    expect(wf.lp_equity_pct).toBe(85)
  })

  it('LP + GP always sum to 100', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    for (const val of [0, 25, 50, 75, 100]) {
      mockUpdateDeal.mockClear()
      fireEvent.change(getInputByLabel('LP Equity (%)'), { target: { value: String(val) } })
      const wf = mockUpdateDeal.mock.calls[0][0].waterfall_config
      expect(wf.lp_equity_pct + wf.gp_equity_pct).toBe(100)
    }
  })

  it('clamps LP > 100 to 100 (GP becomes 0)', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('LP Equity (%)'), { target: { value: '150' } })
    const wf = mockUpdateDeal.mock.calls[0][0].waterfall_config
    expect(wf.lp_equity_pct).toBe(100)
    expect(wf.gp_equity_pct).toBe(0)
  })

  it('clamps LP < 0 to 0 (GP becomes 100)', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('LP Equity (%)'), { target: { value: '-10' } })
    const wf = mockUpdateDeal.mock.calls[0][0].waterfall_config
    expect(wf.lp_equity_pct).toBe(0)
    expect(wf.gp_equity_pct).toBe(100)
  })
})

// ---------------------------------------------------------------------------
// Promote Tier Splits
// ---------------------------------------------------------------------------

describe('Promote tier LP/GP split — linked pair atomicity (bug fix)', () => {
  beforeEach(() => vi.clearAllMocks())

  it('changing tier-0 LP split issues exactly ONE updateDeal call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    const lpSplits = getAllInputsByLabel('LP Split (%)')
    fireEvent.change(lpSplits[0], { target: { value: '75' } })
    expect(mockUpdateDeal).toHaveBeenCalledTimes(1)
  })

  it('tier-0 LP=75 → lp_split=75 and gp_split=25 in same call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getAllInputsByLabel('LP Split (%)')[0], { target: { value: '75' } })
    const tiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(tiers[0].lp_split).toBe(75)
    expect(tiers[0].gp_split).toBe(25)
  })

  it('tier-0 GP=35 → gp_split=35 and lp_split=65 in same call', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getAllInputsByLabel('GP Split (%)')[0], { target: { value: '35' } })
    const tiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(tiers[0].gp_split).toBe(35)
    expect(tiers[0].lp_split).toBe(65)
  })

  it('updating tier-1 does not mutate tier-0 or tier-2', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getAllInputsByLabel('LP Split (%)')[1], { target: { value: '65' } })
    const tiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(tiers[0].lp_split).toBe(80)   // unchanged
    expect(tiers[1].lp_split).toBe(65)   // changed
    expect(tiers[1].gp_split).toBe(35)   // counterpart updated
    expect(tiers[2].lp_split).toBe(60)   // unchanged
  })

  it('updating tier-2 (uncapped) only changes tier-2', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getAllInputsByLabel('LP Split (%)')[2], { target: { value: '55' } })
    const tiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(tiers[2].lp_split).toBe(55)
    expect(tiers[2].gp_split).toBe(45)
    expect(tiers[0].lp_split).toBe(80)   // unchanged
  })

  it('tier split LP + GP always sum to 100', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    for (const val of [0, 30, 50, 70, 100]) {
      mockUpdateDeal.mockClear()
      fireEvent.change(getAllInputsByLabel('LP Split (%)')[0], { target: { value: String(val) } })
      const t = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers[0]
      expect(t.lp_split + t.gp_split).toBe(100)
    }
  })

  it('clamps tier GP split > 100 to 100 (LP becomes 0)', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getAllInputsByLabel('GP Split (%)')[0], { target: { value: '120' } })
    const tiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(tiers[0].gp_split).toBe(100)
    expect(tiers[0].lp_split).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// Single-field waterfall controls (should still work as before)
// ---------------------------------------------------------------------------

describe('Single-field controls', () => {
  beforeEach(() => vi.clearAllMocks())

  it('preferred return input fires updateDeal with new value', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.change(getInputByLabel('Preferred Return (%/yr)'), { target: { value: '10' } })
    expect(mockUpdateDeal).toHaveBeenCalledTimes(1)
    expect(mockUpdateDeal.mock.calls[0][0].waterfall_config.preferred_return).toBe(10)
  })
})

// ---------------------------------------------------------------------------
// Waterfall order summary display
// ---------------------------------------------------------------------------

describe('Waterfall order summary', () => {
  beforeEach(() => vi.clearAllMocks())

  it('displays the correct preferred return rate in the summary', () => {
    setupMock()
    render(<WaterfallConfigPage />)
    // The ordered list contains "LP preferred return (8% simple, cumulative)"
    const listItems = document.querySelectorAll('ol li')
    const prefItem = Array.from(listItems).find((li) => li.textContent?.includes('preferred return'))
    expect(prefItem).toBeTruthy()
    expect(prefItem!.textContent).toMatch(/8%/)
  })

  it('displays all three IRR tier ranges', () => {
    setupMock()
    render(<WaterfallConfigPage />)
    expect(screen.getByText(/IRR 0–14%/)).toBeInTheDocument()
    expect(screen.getByText(/IRR 14–18%/)).toBeInTheDocument()
    expect(screen.getByText(/IRR 18–∞%/)).toBeInTheDocument()
  })

  it('shows the LP/GP equity bar with correct widths', () => {
    setupMock()
    render(<WaterfallConfigPage />)
    const bars = document.querySelectorAll('[style*="width"]')
    const lpBar = Array.from(bars).find((el) => (el as HTMLElement).style.width === '90%')
    const gpBar = Array.from(bars).find((el) => (el as HTMLElement).style.width === '10%')
    expect(lpBar).toBeTruthy()
    expect(gpBar).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------
// Add / Remove tier
// ---------------------------------------------------------------------------

describe('Add and remove tiers', () => {
  beforeEach(() => vi.clearAllMocks())

  it('Add Tier calls updateDeal with one more tier', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    fireEvent.click(screen.getByRole('button', { name: /Add Tier/i }))
    expect(mockUpdateDeal).toHaveBeenCalledTimes(1)
    const newTiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(newTiers).toHaveLength(DEFAULT_WF.tiers.length + 1)
  })

  it('Remove Tier on tier-1 calls updateDeal with one fewer tier', () => {
    const { mockUpdateDeal } = setupMock()
    render(<WaterfallConfigPage />)
    // There are 3 tiers; trash buttons are not disabled (count > 1)
    const trashButtons = screen.getAllByRole('button').filter(
      (btn) => btn.querySelector('svg') && !btn.textContent?.trim()
    )
    fireEvent.click(trashButtons[1])  // remove second tier
    const newTiers = mockUpdateDeal.mock.calls[0][0].waterfall_config.tiers
    expect(newTiers).toHaveLength(DEFAULT_WF.tiers.length - 1)
  })

  it('Remove Tier is disabled when only one tier remains', () => {
    const singleTierWF = { ...DEFAULT_WF, tiers: [DEFAULT_WF.tiers[0]] }
    setupMock(singleTierWF)
    render(<WaterfallConfigPage />)
    const trashBtn = screen.getAllByRole('button').find(
      (btn) => btn.querySelector('svg') && !btn.textContent?.trim()
    )
    expect(trashBtn).toBeDisabled()
  })
})
