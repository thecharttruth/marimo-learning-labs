#!/usr/bin/env node
/** MoLab browser helpers — real Chrome + optional CDP attach (stable Google login). */
import { chromium } from "playwright";
import path from "node:path";

export const profileDir = path.join(
  process.env.HOME,
  "Library/Application Support/molab-playwright",
);

const DEFAULT_CDP = process.env.MOLAB_CDP_URL || "http://127.0.0.1:9222";

async function cdpReachable(url) {
  try {
    const res = await fetch(`${url.replace(/\/$/, "")}/json/version`);
    return res.ok;
  } catch {
    return false;
  }
}

/** @returns {{ context: import('playwright').BrowserContext, browser: import('playwright').Browser | null, mode: 'cdp' | 'launch' }} */
export async function connectMolabBrowser({ cdpUrl = DEFAULT_CDP, preferCdp = true } = {}) {
  if (preferCdp && (await cdpReachable(cdpUrl))) {
    const browser = await chromium.connectOverCDP(cdpUrl);
    const context = browser.contexts()[0];
    if (!context) {
      throw new Error(`CDP connected at ${cdpUrl} but no browser context found.`);
    }
    return { browser, context, mode: "cdp" };
  }

  const context = await chromium.launchPersistentContext(profileDir, {
    channel: "chrome",
    headless: false,
    viewport: { width: 1400, height: 900 },
    ignoreDefaultArgs: ["--enable-automation"],
    args: ["--disable-blink-features=AutomationControlled"],
  });
  return { browser: null, context, mode: "launch" };
}

export async function closeMolabBrowser(session) {
  if (session.mode === "launch") {
    await session.context.close();
  } else if (session.browser) {
    await session.browser.close();
  }
}

export async function getOrCreateMolabPage(context) {
  for (const page of context.pages()) {
    if (page.url().includes("molab.marimo.io")) {
      return page;
    }
  }
  const page = context.pages()[0] ?? (await context.newPage());
  return page;
}
