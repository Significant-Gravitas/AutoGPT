import { expect, test } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MainMarkeplacePage } from './MainMarketplacePage'
 
test('MainMarketplacePage', () => {
  render(<MainMarkeplacePage />)
  expect(screen.getByText('Featured Agents')).toBeDefined()
})