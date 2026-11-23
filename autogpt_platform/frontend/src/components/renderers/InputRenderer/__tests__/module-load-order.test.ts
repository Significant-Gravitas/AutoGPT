import { readdirSync } from "node:fs";
import path from "node:path";
import { beforeAll, describe, expect, it, vi } from "vitest";

// registry/Form.tsx builds the form when it loads (`export default
// generateForm()`), reading every field component. If loading any other
// renderer module first can lead back to Form before a field has initialised,
// Form throws "Cannot access '<field>' before initialization". The production
// server's start-up preload loads modules in module-id order, so whichever
// module that happens to be decides it, and a dependency change can reorder it.
const RENDERER_ROOT = path.resolve(__dirname, "..");

// Compiling the renderer graph from a cold cache takes far longer than running
// it, and a slow runner can take minutes. It happens once, before the cases;
// each case then only runs the modules again, in a fresh registry.
const WARM_UP_TIMEOUT_MS = 300_000;
const CASE_TIMEOUT_MS = 60_000;

const modules = readdirSync(RENDERER_ROOT, {
  recursive: true,
  encoding: "utf8",
})
  .map((file) => file.split(path.sep).join("/"))
  .filter((file) => /\.tsx?$/.test(file))
  .filter((file) => !/\.(test|stories)\.tsx?$/.test(file))
  .filter((file) => !/(^|\/)(__tests__|stories)\//.test(file))
  .sort();

describe("InputRenderer modules", () => {
  beforeAll(async () => {
    // Through the entry point the app uses, which loads every module in the
    // order that works, so this warms the transform cache without testing it.
    await import(path.join(RENDERER_ROOT, "FormRenderer.tsx"));
  }, WARM_UP_TIMEOUT_MS);

  it("finds the renderer's modules", () => {
    expect(modules).toContain("registry/Form.tsx");
  });

  it.each(modules)(
    "%s loads when it is the first module imported",
    async (file) => {
      vi.resetModules();
      await expect(
        import(path.join(RENDERER_ROOT, file)),
      ).resolves.toBeDefined();
    },
    CASE_TIMEOUT_MS,
  );
});
