#!/usr/bin/env node
/**
 * Create MoLab workspace notebooks mirrored from GitHub (auto-sync on push).
 * Reads manifest JSON from molab-profile.py mirror command.
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
  let manifest = path.join(__dirname, "..", ".molab-import", "mirror-manifest.json");
  let results = path.join(__dirname, "..", ".molab-import", "mirror-results.json");
  let loginOnly = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--manifest" && args[i + 1]) manifest = path.resolve(args[++i]);
    else if (args[i] === "--results" && args[i + 1]) results = path.resolve(args[++i]);
    else if (args[i] === "--login-only") loginOnly = true;
  }
  return { manifest, results, loginOnly };
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
  throw new Error(`Timed out waiting for MoLab home. Run: ./scripts/molab-github-mirror.sh login`);
}

async function resolveCatalogUrl(page, entry) {
  const slug = entry.slug;
  const title = entry.title || slug;
  const slugPat = slug.replace(/-/g, "[-_ ]");
  const link = page
    .getByRole("link", { name: new RegExp(slugPat, "i") })
    .or(page.getByRole("link", { name: new RegExp(title.split(" ").slice(0, 2).join("|"), "i") }));
  await link.first().waitFor({ state: "visible", timeout: 60_000 });
  const href = await link.first().getAttribute("href");
  if (!href || !href.includes("/notebooks/nb_")) {
    throw new Error(`Could not resolve catalog URL for ${slug}`);
  }
  return href.split("?")[0].replace(/\/$/, "");
}

async function mirrorOne(page, entry) {
  const label = entry.slug || entry.github_url;
  console.log(`Mirroring ${label}...`);
  console.log(`  GitHub: ${entry.github_url}`);

  await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
  await waitForMolabHome(page);
  await page.waitForTimeout(800);

  try {
    await page.getByRole("button", { name: /more notebook creation options/i }).click();
    await page.waitForTimeout(400);
    await page
      .getByRole("menuitem")
      .filter({ hasText: /mirror from github/i })
      .first()
      .click({ timeout: 30_000 });

    const dialog = page.getByRole("dialog");
    await dialog.waitFor({ state: "visible", timeout: 60_000 });
    const urlField = dialog.locator('input[type="url"]').or(dialog.getByRole("textbox")).first();
    await urlField.fill(entry.github_url);
    await dialog.getByRole("button", { name: "Add to workspace" }).click({ timeout: 120_000 });
    await page.waitForTimeout(2500);
  } catch (err) {
    console.log(`  mirror dialog skipped/failed (${err.message?.slice(0, 80)}); resolving from home list`);
    await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
    await waitForMolabHome(page);
  }

  const catalogUrl = await resolveCatalogUrl(page, entry);
  console.log(`  -> ${catalogUrl}`);
  return {
    slug: entry.slug,
    title: entry.title,
    github_url: entry.github_url,
    catalog_url: catalogUrl,
    cloud_kind: "github_sync",
  };
}

async function main() {
  const { manifest, results, loginOnly } = await parseArgs();
  const session = await connectMolabBrowser({ preferCdp: true });
  let page = await getOrCreateMolabPage(session.context);

  try {
    await page.goto(MOLAB_HOME, { waitUntil: "domcontentloaded" });
    await waitForMolabHome(page);
    console.log("MoLab home ready.");

    if (loginOnly) {
      console.log("Login confirmed. Run: ./scripts/molab-github-mirror.sh mirror");
      return;
    }

    const payload = JSON.parse(await readFile(manifest, "utf8"));
    const notebooks = payload.notebooks || [];
    if (!notebooks.length) {
      console.error(`No notebooks in manifest: ${manifest}`);
      process.exit(1);
    }

    const mirrored = [];
    for (const entry of notebooks) {
      if (page.isClosed()) {
        page = await session.context.newPage();
      }
      mirrored.push(await mirrorOne(page, entry));
      await page.waitForTimeout(1000);
    }

    const out = { mirrored_at: new Date().toISOString(), notebooks: mirrored };
    await writeFile(results, JSON.stringify(out, null, 2) + "\n");
    console.log(`\nWrote ${results}`);
    for (const row of mirrored) {
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
