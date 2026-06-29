#!/usr/bin/env node
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  closeMolabBrowser,
  connectMolabBrowser,
  getOrCreateMolabPage,
} from "./molab-browser.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(__dirname, "..", ".molab-import");

async function dump(page, label) {
  console.log(`\n=== ${label} ===`);
  console.log("URL:", page.url());

  const buttons = await page.locator("button:visible").allTextContents();
  console.log("Buttons:", buttons.filter(Boolean));

  const menuitems = await page.locator('[role="menuitem"]:visible').allTextContents();
  console.log("Menuitems:", menuitems.filter(Boolean));

  const menu = await page.locator('[role="menu"]:visible').allTextContents();
  console.log("Menus:", menu.filter(Boolean));

  const popovers = await page.locator('[data-radix-popper-content-wrapper]:visible').allTextContents();
  console.log("Popovers:", popovers.filter(Boolean).slice(0, 5));

  const haspopup = await page.locator('[aria-haspopup="menu"]:visible').evaluateAll((els) =>
    els.map((el) => ({
      tag: el.tagName,
      text: el.textContent?.trim().slice(0, 80),
      aria: el.getAttribute("aria-label"),
      expanded: el.getAttribute("aria-expanded"),
    })),
  );
  console.log("aria-haspopup=menu:", haspopup);
}

async function main() {
  const session = await connectMolabBrowser({ preferCdp: true });
  const page = await getOrCreateMolabPage(session.context);
  await page.goto("https://molab.marimo.io/notebooks", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);
  await dump(page, "HOME");

  await page.screenshot({ path: path.join(OUT, "debug-home.png"), fullPage: true });

  const newBtn = page.getByRole("button", { name: /^new notebook$/i });
  console.log("\nNew notebook visible:", await newBtn.isVisible());
  await newBtn.click();
  await page.waitForTimeout(1000);
  await dump(page, "AFTER NEW NOTEBOOK CLICK");
  await page.screenshot({ path: path.join(OUT, "debug-after-new.png"), fullPage: true });

  const triggers = page.locator('[aria-haspopup="menu"]:visible');
  const n = await triggers.count();
  for (let i = 0; i < n; i++) {
    const t = triggers.nth(i);
    const text = await t.textContent();
    console.log(`\nClicking haspopup[${i}]:`, text?.trim().slice(0, 60));
    await t.click();
    await page.waitForTimeout(800);
    await dump(page, `AFTER HASPOPUP[${i}]`);
    await page.screenshot({ path: path.join(OUT, `debug-haspopup-${i}.png`), fullPage: true });
    await page.keyboard.press("Escape");
    await page.waitForTimeout(300);
  }

  await closeMolabBrowser(session);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
