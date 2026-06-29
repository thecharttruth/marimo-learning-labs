#!/usr/bin/env node
/**
 * Import notebooks into MoLab profile (catalog nb_* URLs).
 * Reads a manifest JSON produced by molab-profile.py.
 */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  closeMolabBrowser,
  connectMolabBrowser,
  getOrCreateMolabPage,
} from "./molab-browser.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MOLAB_HOME = "https://molab.marimo.io/notebooks";

async function parseArgs() {
  const args = process.argv.slice(2);
  let manifest = path.join(__dirname, "..", ".molab-import", "manifest.json");
  let results = path.join(__dirname, "..", ".molab-import", "results.json");
  let debug = false;
  let loginOnly = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--manifest" && args[i + 1]) manifest = path.resolve(args[++i]);
    else if (args[i] === "--results" && args[i + 1]) results = path.resolve(args[++i]);
    else if (args[i] === "--debug") debug = true;
    else if (args[i] === "--login-only") loginOnly = true;
  }
  return { manifest, results, debug, loginOnly };
}

async function waitForMolabHome(page, timeoutMs = 300_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const url = page.url();
    if (url.includes("molab.marimo.io/notebooks")) {
      const ready = await page
        .getByRole("button", { name: /new notebook/i })
        .isVisible()
        .catch(() => false);
      if (ready) return;
    }
    if (url.includes("accounts.google.com")) {
      console.log("Complete Google sign-in in the Chrome window...");
    }
    await page.waitForTimeout(1500);
  }
  throw new Error(
    `Timed out waiting for MoLab home (${MOLAB_HOME}). Run: ./scripts/molab-profile.sh login`,
  );
}

async function debugUi(page) {
  const buttons = await page.locator("button").allTextContents();
  console.log("Visible buttons:", buttons.filter(Boolean).slice(0, 30));
  await page.screenshot({ path: path.join(__dirname, "..", ".molab-import", "debug.png"), fullPage: true });
  console.log("Screenshot: .molab-import/debug.png");
}

async function openImportUploadDialog(page) {
  await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
  await waitForMolabHome(page);
  await page.waitForTimeout(1000);

  const importBtn = page.getByRole("button", { name: /^import notebook$/i });
  if (await importBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
    await importBtn.click();
    return;
  }

  // MoLab home: chevron beside "New notebook" opens creation menu (Import lives here).
  await page
    .getByRole("button", { name: /more notebook creation options/i })
    .click();
  await page.waitForTimeout(400);

  const importItem = page
    .getByRole("menuitem")
    .filter({ hasText: /upload a notebook/i })
    .or(page.getByRole("menuitem", { name: /^import$/i }));
  await importItem.first().click({ timeout: 60_000 });
}

async function uploadNotebook(page, filePath) {
  const dialog = page.getByRole("dialog");
  await dialog.waitFor({ state: "visible", timeout: 60_000 });

  const uploadTab = dialog.getByRole("tab", { name: /upload/i });
  if (await uploadTab.isVisible({ timeout: 2000 }).catch(() => false)) {
    await uploadTab.click();
  }

  const fileInput = dialog.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);

  const submit = dialog
    .getByRole("button", { name: /^import$/i })
    .or(dialog.getByRole("button", { name: /upload|open|create/i }));
  await submit.first().click({ timeout: 120_000 });

  await page.waitForURL(/molab\.marimo\.io\/notebooks\/nb_/, { timeout: 180_000 });
  return page.url().replace(/\/$/, "");
}

async function importOne(page, entry, debug) {
  const label = entry.slug || path.basename(entry.staging_path, ".py");
  console.log(`Importing ${label}...`);
  await openImportUploadDialog(page);
  if (debug) await debugUi(page);
  const url = await uploadNotebook(page, entry.staging_path);
  console.log(`  -> ${url}`);
  return { slug: entry.slug, title: entry.title, catalog_url: url };
}

async function main() {
  const { manifest, results, debug, loginOnly } = await parseArgs();
  const session = await connectMolabBrowser({ preferCdp: true });
  const page = await getOrCreateMolabPage(session.context);

  try {
    await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
    await waitForMolabHome(page);
    console.log("MoLab home ready.");

    if (loginOnly) {
      console.log("Login confirmed. Run: ./scripts/molab-profile.sh publish");
      return;
    }

    const payload = JSON.parse(await readFile(manifest, "utf8"));
    const notebooks = payload.notebooks || [];
    if (!notebooks.length) {
      console.error(`No notebooks in manifest: ${manifest}`);
      process.exit(1);
    }

    const imported = [];
    for (const entry of notebooks) {
      imported.push(await importOne(page, entry, debug));
    }

    const out = { imported_at: new Date().toISOString(), notebooks: imported };
    await writeFile(results, JSON.stringify(out, null, 2) + "\n");
    console.log(`\nWrote ${results}`);
    for (const row of imported) {
      console.log(`${row.slug}: ${row.catalog_url}`);
    }
  } finally {
    await closeMolabBrowser(session);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
