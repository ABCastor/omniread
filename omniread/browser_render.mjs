#!/usr/bin/env node
// Thin Tier-2 adapter. Playwright and its isolated Chromium are installed by the operator.
// Never add channel: 'chrome' or a system executablePath here.
import { createRequire } from 'node:module';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const args = process.argv.slice(2);
const url = args[0];

function option(name) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : null;
}

function options(name) {
  const values = [];
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === name && args[index + 1]) values.push(args[index + 1]);
  }
  return values;
}

const root = option('--playwright-root');
const userDataDir = option('--user-data-dir');
const login = args.includes('--login');
const loginTimeout = Number(option('--login-timeout') || 600);
if (!url || !root) {
  console.error(
    'usage: browser_render.mjs <url> --playwright-root <bundle-directory> ' +
    '[--user-data-dir <dir>] [--login] [--login-timeout <seconds>]'
  );
  process.exit(2);
}
if (login && !userDataDir) {
  console.error('--login requires --user-data-dir');
  process.exit(2);
}
if (login && (!Number.isFinite(loginTimeout) || loginTimeout <= 0)) {
  console.error('--login-timeout must be a positive number of seconds');
  process.exit(2);
}

const require = createRequire(path.join(root, 'package.json'));
const { chromium } = require('playwright');
const extensions = options('--extension');
const startedAt = Date.now();
const viewport = { width: 1280, height: 1800 };
let browser = null;
let context = null;
let profile = null;

try {
  if (userDataDir || extensions.length) {
    profile = userDataDir || (await mkdtemp(path.join(os.tmpdir(), 'omniread-browser-')));
    const extensionArg = extensions.join(',');
    const launchArgs = extensions.length
      ? [
          `--disable-extensions-except=${extensionArg}`,
          `--load-extension=${extensionArg}`,
        ]
      : [];
    context = await chromium.launchPersistentContext(profile, {
      headless: !login,
      viewport,
      args: launchArgs,
    });
  } else {
    browser = await chromium.launch({ headless: true });
    context = await browser.newContext({ viewport });
  }
  const page = context.pages()[0] || (await context.newPage());
  let response = null;
  let finalUrl = url;
  page.on('framenavigated', frame => {
    if (frame === page.mainFrame()) finalUrl = frame.url();
  });
  try {
    response = await page.goto(url, {
      waitUntil: login ? 'domcontentloaded' : 'networkidle',
      timeout: 45000,
    });
  } catch {
    response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 });
  }
  finalUrl = page.url();
  if (login) {
    const completion = await new Promise(resolve => {
      let finished = false;
      let timer = null;
      const finish = reason => {
        if (!finished) {
          finished = true;
          if (timer) clearTimeout(timer);
          resolve(reason);
        }
      };
      page.once('close', () => finish('window-closed'));
      context.once('close', () => finish('window-closed'));
      timer = setTimeout(() => finish('timeout'), loginTimeout * 1000);
    });
    if (!page.isClosed()) finalUrl = page.url();
    process.stdout.write(JSON.stringify({
      final_url: finalUrl,
      completion,
      elapsed_ms: Date.now() - startedAt,
    }));
  } else {
    await page.waitForTimeout(1500);
    process.stdout.write(JSON.stringify({
      html: await page.content(),
      status: response ? response.status() : 599,
      final_url: finalUrl,
      elapsed_ms: Date.now() - startedAt,
    }));
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  else if (context) {
    try { await context.close(); } catch { /* user already closed the window */ }
  }
  if (profile && !userDataDir) await rm(profile, { recursive: true, force: true });
}
