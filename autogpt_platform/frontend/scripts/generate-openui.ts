import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { getSystemPrompt } from "../src/lib/openui/catalog";

const target = resolve("../backend/backend/copilot/tools/openui_library.txt");
const prompt = `${getSystemPrompt()}\n`;

if (process.argv.includes("--check")) {
  if (readFileSync(target, "utf8") !== prompt) {
    throw new Error("OpenUI catalog changed. Run pnpm generate:openui.");
  }
} else {
  writeFileSync(target, prompt);
}
