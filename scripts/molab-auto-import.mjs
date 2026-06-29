#!/usr/bin/env node
/** @deprecated Use: ./scripts/molab-profile.sh publish */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
spawnSync("bash", ["./scripts/molab-profile.sh", "publish", ...process.argv.slice(2)], {
  cwd: root,
  stdio: "inherit",
});
