import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { getSystemPrompt } from "../catalog";

describe("Copilot OpenUI catalog contract", () => {
  it("gives the backend model exactly the component library the frontend can render", () => {
    const prompt = readFileSync(
      "../backend/backend/copilot/tools/openui_library.txt",
      "utf8",
    );
    expect(prompt).toBe(`${getSystemPrompt()}\n`);
  });
});
