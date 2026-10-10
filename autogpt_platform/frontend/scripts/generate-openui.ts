import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { getSystemPrompt } from "../src/lib/openui/catalog";
import { buildOpenUIValidator } from "./build-openui-validator";

async function main() {
  const artifacts = [
    { name: "openui_library.txt", text: `${getSystemPrompt()}\n` },
    ...(await buildOpenUIValidator()),
  ];
  for (const artifact of artifacts) {
    const target = resolve("../backend/backend/copilot/tools", artifact.name);
    if (process.argv.includes("--check")) {
      if (readFileSync(target, "utf8") !== artifact.text) {
        throw new Error(
          `OpenUI artifact ${artifact.name} changed. Run pnpm generate:openui.`,
        );
      }
    } else {
      writeFileSync(target, artifact.text);
    }
  }
}

void main();
