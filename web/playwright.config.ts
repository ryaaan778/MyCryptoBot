import { defineConfig } from '@playwright/test'

/**
 * The backend serves the built frontend from `web/dist`, so these tests run
 * against the real production path: one server, real WebSocket, real engine.
 *
 * `executablePath` points at the Chromium already present in this environment
 * rather than downloading a matching revision.
 */

const PORT = Number(process.env.JOJO_E2E_PORT ?? 8012)
const BASE_URL = process.env.JOJO_E2E_URL ?? `http://127.0.0.1:${PORT}`

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 25_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    baseURL: BASE_URL,
    viewport: { width: 1600, height: 950 },
    launchOptions: {
      executablePath: '/opt/pw-browsers/chromium',
      args: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        // Software GL: the sandbox has no GPU, but the world must still render.
        '--use-gl=swiftshader',
        '--enable-unsafe-swiftshader',
        '--disable-gpu-sandbox',
      ],
    },
    screenshot: 'only-on-failure',
    trace: 'off',
  },
})
