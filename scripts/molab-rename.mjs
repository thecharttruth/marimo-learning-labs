#!/usr/bin/env node
/**
 * Rename MoLab workspace notebooks to registry titles (not "notebook.py").
 */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { closeMolabBrowser, connectMolabBrowser, getOrCreateMolabPage } from "./molab-browser.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MOLAB_HOME = "https://molab.marimo.io/notebooks";

async function parseArgs() {
  const args = process.argv.slice(2);
  let manifest = path.join(__dirname, "..", ".molab-import", "rename-manifest.json");
  let loginOnly = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--manifest" && args[i + 1]) manifest = path.resolve(args[++i]);
    else if (args[i] === "--login-only") loginOnly = true;
  }
  return { manifest, loginOnly };
}

async function waitForMolabHome(page) {
  await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
  for (let i = 0; i < 120; i++) {
    if (
      page.url().includes("molab.marimo.io/notebooks") &&
      (await page.getByRole("button", { name: /new notebook/i }).isVisible().catch(() => false))
    ) {
      return;
    }
    await page.waitForTimeout(1000);
  }
  throw new Error("MoLab home not ready");
}

async function renameOne(page, entry) {
  const { title, catalog_url, slug, match_names = [] } = entry;
  console.log(`Renaming -> "${title}" (${slug})`);

  await waitForMolabHome(page);
  await page.waitForTimeout(800);

  const candidates = [title, slug, "notebook.py", ...match_names];
  let card = null;
  for (const name of candidates) {
    const loc = page.getByRole("link", { name: new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`, "i") });
    if (await loc.count()) {
      card = loc.first();
      break;
    }
  }

  if (!card && catalog_url) {
    const nbId = catalog_url.split("/").pop();
    card = page.locator(`a[href*="${nbId}"]`).first();
  }

  if (!card || !(await card.count())) {
    throw new Error(`Could not find notebook card for ${slug}`);
  }

  await card.hover();
  await page.waitForTimeout(300);

  const row = card.locator("xpath=ancestor::*[self::div or self::li][1]");
  const menuBtn = row
    .locator('[aria-haspopup="menu"]')
    .or(row.getByRole("button", { name: /more|options|menu/i }))
    .last();
  if (await menuBtn.count()) {
    await menuBtn.click();
  } else {
    await card.click({ button: "right" });
  }
  await page.waitForTimeout(400);

  const renameItem = page
    .getByRole("menuitem", { name: /^rename$/i })
    .or(page.getByRole("menuitem").filter({ hasText: /^rename/i }));
  await renameItem.first().click({ timeout: 10_000 });

  const field = page
    .getByRole("textbox")
    .or(page.locator('input[type="text"]'))
    .filter({ hasNot: page.locator('[type="url"]') })
    .last();
  await field.fill(title);

  const save = page
    .getByRole("button", { name: /^save$/i })
    .or(page.getByRole("button", { name: /^rename$/i }))
    .or(page.getByRole("button", { name: /^ok$/i }));
  await save.first().click({ timeout: 10_000 });
  await page.waitForTimeout(1000);

  console.log(`  done`);
}

async function main() {
  const { manifest, loginOnly } = await parseArgs();
  const session = await connectMolabBrowser({ preferCdp: true });
  const page = await getOrCreateMolabPage(session.context);

  try {
    await waitForMolabHome(page);
    if (loginOnly) {
      console.log("Login OK. Run: ./scripts/molab-rename.sh");
      return;
    }

    const payload = JSON.parse(await readFile(manifest, "utf8"));
    for (const entry of payload.notebooks) {
      await renameOne(page, entry);
    }
  } finally {
    await closeMolabBrowser(session);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
