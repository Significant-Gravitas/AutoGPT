// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { buildOpenUIValidator } from "./build-openui-validator";

it("ships the exact canonical parser and schemas to the backend", async () => {
  for (const artifact of await buildOpenUIValidator()) {
    expect(
      readFileSync(
        resolve("../backend/backend/copilot/tools", artifact.name),
        "utf8",
      ),
    ).toBe(artifact.text);
  }
});
