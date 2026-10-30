// Run SQL against the running app's database, as its administrator. For
// the one thing no screen can do: put an install into the state an older
// version left it in. Uses the Python and the PostgreSQL driver the app
// ships, so the test machine needs neither.

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { dataDir } from "./config";
import { platform } from "./platform";

const SCRIPT = [
  "import os, sys, psycopg2",
  "connection = psycopg2.connect(host='127.0.0.1', port=int(os.environ['E2E_PG_PORT']),",
  "    user='postgres', password=os.environ['E2E_PG_PASSWORD'], dbname='postgres')",
  "connection.autocommit = True",
  "cursor = connection.cursor()",
  "cursor.execute(sys.stdin.read())",
  "print(cursor.rowcount)",
].join("\n");

/** Returns the number of rows the statement touched. */
export function executeSql(statement: string): Promise<number> {
  const config = path.join(dataDir(), "config");
  const ports = JSON.parse(fs.readFileSync(path.join(config, "ports.json"), "utf8"));
  const python = path.join(
    platform.runtimeDir(),
    "python",
    process.platform === "win32" ? "python.exe" : path.join("bin", "python3"),
  );
  return new Promise((resolve, reject) => {
    // -B: nothing may be written into the installed app, bytecode included.
    const child = spawn(python, ["-B", "-c", SCRIPT], {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        E2E_PG_PORT: String(ports.postgres),
        E2E_PG_PASSWORD: readEnvFile(path.join(config, "runtime.env")).POSTGRES_PASSWORD,
      },
    });
    let output = "";
    child.stdout.on("data", (chunk) => (output += chunk));
    child.stderr.on("data", (chunk) => (output += chunk));
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0) resolve(Number(output.trim()));
      else reject(new Error(`the SQL statement failed (exit ${code}):\n${output}`));
    });
    child.stdin.end(statement);
  });
}

/** KEY=VALUE lines, as runtime/autogpt_desktop/settings.py reads them. */
function readEnvFile(file: string): Record<string, string> {
  const values: Record<string, string> = {};
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const trimmed = line.trim();
    const separator = trimmed.indexOf("=");
    if (!trimmed || trimmed.startsWith("#") || separator < 0) continue;
    values[trimmed.slice(0, separator).trim()] = trimmed.slice(separator + 1).trim();
  }
  return values;
}
