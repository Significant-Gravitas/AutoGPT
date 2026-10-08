import { readFileSync, writeFileSync } from "node:fs";
import { getSystemPrompt } from "../src/lib/openui/catalog.ts";

const target = new URL(
  "../../backend/backend/copilot/tools/openui_library.txt",
  import.meta.url,
);
const prompt = `${getSystemPrompt()}\n`;

if (process.argv.includes("--check")) {
  if (readFileSync(target, "utf8") !== prompt) {
    throw new Error(
      "OpenUI catalog changed. Run node scripts/generate-openui.mts.",
    );
  }
} else {
  writeFileSync(target, prompt);
}
