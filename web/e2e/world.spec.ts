/**
 * End-to-end verification against a live backend.
 *
 * These assert the two things that matter most: that the world renders real
 * trading data, and that the desk stays fully visible when it can't.
 *
 * A note on `?mode=3d`: this sandbox has no GPU, so Chromium falls back to
 * software rasterisation and the world runs at roughly one frame per second.
 * The app is *supposed* to notice that and switch to the 2D dashboard — which
 * is itself covered below. The world tests therefore pin the 3D view on
 * explicitly, so they exercise the world rather than the safeguard.
 */

import { expect, test, type Page } from '@playwright/test'
import { mkdirSync } from 'node:fs'

const SHOTS = 'screenshots'
mkdirSync(SHOTS, { recursive: true })

/** Software rendering is slow; the world needs generous time here. */
const WORLD = '/?mode=3d'

async function waitForLive(page: Page): Promise<void> {
  await expect(page.locator('.topbar__title')).toContainText('JOJO TRADING COMMAND CENTER')
  await expect(page.locator('.badge--live')).toContainText('LIVE', { timeout: 45_000 })
}

/** Let the world compile geometry and draw a few frames before capturing. */
async function settle(page: Page, ms = 2500): Promise<void> {
  await page.waitForTimeout(ms)
}

test.describe('voxel world', () => {
  test.describe.configure({ timeout: 180_000 })

  test('connects, renders the world, and shows live trading data', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text())
    })

    await page.goto(WORLD)
    await waitForLive(page)

    const canvas = page.locator('canvas')
    await expect(canvas).toBeVisible()
    const box = await canvas.boundingBox()
    expect(box!.width).toBeGreaterThan(600)

    for (const name of ['JONATHAN', 'JOSEPH', 'JOTARO', 'JOLYAN', 'KIRA']) {
      await expect(page.locator('.roster__name', { hasText: name })).toBeVisible()
    }

    // Paper mode must be unmistakable.
    await expect(page.locator('.badge--paper')).toContainText('PAPER TRADING')

    // Equity is a real number, not a placeholder.
    const equity = await page.locator('.stat', { hasText: 'Equity' })
      .locator('.stat__value').first().textContent()
    expect(equity).toMatch(/\$[\d,.KM]+/)

    await settle(page, 4000)
    await page.screenshot({ path: `${SHOTS}/01-world.png` })

    expect(errors, `console errors: ${errors.join(' | ')}`).toHaveLength(0)
  })

  test('camera presets fly to the headquarters and market room', async ({ page }) => {
    await page.goto(WORLD)
    await waitForLive(page)
    await settle(page, 2000)

    await page.locator('.camerabar__button', { hasText: 'JOJO' }).first().click()
    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/02-headquarters.png` })

    await page.locator('.camerabar__button', { hasText: 'MARKET' }).click()
    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/03-market-room.png` })
  })

  test('camera presets fly to the trading floor and risk center', async ({ page }) => {
    await page.goto(WORLD)
    await waitForLive(page)
    await settle(page, 2000)

    await page.locator('.camerabar__button', { hasText: 'TRADING FLOOR' }).click()
    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/04-trading-floor.png` })

    await page.locator('.camerabar__button', { hasText: 'RISK CENTER' }).click()
    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/05-risk-center.png` })
  })

  test('selecting a bot opens its district and detail panel', async ({ page }) => {
    await page.goto(WORLD)
    await waitForLive(page)
    await settle(page, 2000)

    await page.locator('.roster__item', { hasText: 'JOTARO' }).click()

    const panel = page.locator('.botpanel')
    await expect(panel).toBeVisible()
    await expect(panel.locator('.botpanel__name')).toHaveText('JOTARO')
    // The panel shows the bot's real configuration, not a stub.
    await expect(panel).toContainText('SCALPING')
    await expect(panel).toContainText('BTCUSDT')

    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/06-bot-district.png` })

    await page.locator('.roster__item', { hasText: 'KIRA' }).click()
    await expect(panel.locator('.botpanel__name')).toHaveText('KIRA')
    await expect(panel).toContainText('MEAN REVERSION')
    await settle(page, 3500)
    await page.screenshot({ path: `${SHOTS}/07-kira-district.png` })
  })

  test('the event log receives live trading events', async ({ page }) => {
    await page.goto(WORLD)
    await waitForLive(page)

    const rows = page.locator('.eventlog__row')
    await expect(rows.first()).toBeVisible({ timeout: 60_000 })

    // Wait for real order flow, not just the startup banner.
    await expect(async () => {
      const text = await page.locator('.eventlog__list').innerText()
      expect(text).toMatch(/opened|submitted|detected|closed/i)
    }).toPass({ timeout: 60_000 })

    await page.screenshot({ path: `${SHOTS}/08-event-log.png` })
  })

  test('data tables expose orders, trades and the audit trail', async ({ page }) => {
    await page.goto(WORLD)
    await waitForLive(page)

    await page.locator('.iconbutton', { hasText: 'DATA TABLES' }).click()
    await expect(page.locator('.tables')).toBeVisible()

    for (const tab of ['ORDERS', 'TRADES', 'EVENT LOG', 'AUDIT TRAIL']) {
      await page.locator('.tables__tabs .camerabar__button', { hasText: tab }).click()
      await page.waitForTimeout(500)
    }

    // The audit trail must contain the system's own records.
    await expect(page.locator('.tables__body')).toContainText(/start|open_position|bot\./i, {
      timeout: 25_000,
    })
    await page.screenshot({ path: `${SHOTS}/09-data-tables.png` })
  })
})

test.describe('fallback', () => {
  test('?mode=2d renders the full dashboard with no WebGL context', async ({ page }) => {
    await page.goto('/?mode=2d')

    const fallback = page.locator('[data-testid="fallback-2d"]')
    await expect(fallback).toBeVisible()
    await expect(page.locator('canvas')).toHaveCount(0)

    await expect(fallback).toContainText('SIMPLIFIED VIEW')
    await expect(fallback).toContainText('PAPER TRADING')

    // Every trading surface the world offers must be present here too.
    for (const heading of [
      'Total Equity', 'Total P&L', 'Risk Status', 'Active Bots',
      'Market', 'Bots', 'Open Positions', 'Recent Trades', 'Orders', 'Event Log',
    ]) {
      await expect(fallback).toContainText(heading, { ignoreCase: true })
    }

    for (const name of ['JONATHAN', 'JOSEPH', 'JOTARO', 'JOLYAN', 'KIRA']) {
      await expect(fallback).toContainText(name)
    }

    // Bot commands are reachable without the 3D view.
    await expect(fallback.locator('button', { hasText: 'PAUSE' }).first()).toBeVisible()

    await page.waitForTimeout(2500)
    await page.screenshot({ path: `${SHOTS}/10-fallback-2d.png`, fullPage: true })
  })

  test('an unusable frame rate degrades and then falls back on its own', async ({ page }) => {
    // This machine has no GPU, so the world genuinely cannot hold a frame rate.
    // That makes it a real test of the safeguard rather than a simulated one:
    // loading the world with no override must end up on the 2D dashboard.
    test.setTimeout(120_000)

    await page.goto('/')
    await expect(page.locator('.topbar__title')).toContainText('JOJO TRADING COMMAND CENTER')

    await expect(page.locator('[data-testid="fallback-2d"]')).toBeVisible({ timeout: 90_000 })
    await expect(page.locator('canvas')).toHaveCount(0)
    await expect(page.locator('[data-testid="fallback-2d"]')).toContainText(
      'could not hold a usable frame rate',
    )

    // The point of the fallback: the data is all still here.
    await expect(page.locator('[data-testid="fallback-2d"]')).toContainText('Total Equity')
    await page.screenshot({ path: `${SHOTS}/12-auto-fallback.png` })
  })
})

test.describe('emergency stop', () => {
  test.describe.configure({ timeout: 180_000 })

  test('halts the desk from the UI and propagates to the backend', async ({ page, request }) => {
    await page.goto(WORLD)
    await waitForLive(page)
    await page.waitForTimeout(5000)

    page.once('dialog', (dialog) => dialog.accept())
    await page.locator('button', { hasText: 'EMERGENCY STOP' }).click()

    // The world enters its emergency state...
    await expect(page.locator('.banner--emergency')).toBeVisible({ timeout: 30_000 })
    await expect(page.locator('.stat__value', { hasText: 'CRITICAL' })).toBeVisible()

    // ...and the backend really did halt and flatten.
    const risk = await (await request.get('/api/risk')).json()
    expect(risk.emergency_stop).toBe(true)
    expect(risk.open_positions).toBe(0)

    const bots = await (await request.get('/api/bots')).json()
    expect(bots.every((bot: { status: string }) => bot.status === 'HALTED')).toBe(true)

    await page.waitForTimeout(3000)
    await page.screenshot({ path: `${SHOTS}/11-emergency-stop.png` })

    // Resume puts it back.
    await page.locator('button', { hasText: 'RESUME SYSTEM' }).click()
    await expect(page.locator('.banner--emergency')).toHaveCount(0, { timeout: 30_000 })
    const after = await (await request.get('/api/risk')).json()
    expect(after.emergency_stop).toBe(false)
  })
})
