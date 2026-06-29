#!/usr/bin/env node
/** @deprecated Use: ./scripts/molab-profile.sh login */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
spawnSync("bash", ["./scripts/molab-profile.sh", "login"], { cwd: root, stdio: "inherit" });
