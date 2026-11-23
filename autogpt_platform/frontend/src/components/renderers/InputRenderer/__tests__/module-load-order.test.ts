import { readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";

// registry/Form.tsx builds the form when it loads (`export default
// generateForm()`), reading every field component. If loading any other
// renderer module first can lead back to Form before a field has initialised,
// Form throws "Cannot access '<field>' before initialization". The production
// server's start-up preload loads modules in module-id order, so whichever
// module that happens to be decides it, and a dependency change can reorder it.
const RENDERER_ROOT = path.resolve(__dirname, "..");

const modules = readdirSync(RENDERER_ROOT, {
  recursive: true,
  encoding: "utf8",
})
  .filter((file) => /\.tsx?$/.test(file))
  .filter((file) => !/\.(test|stories)\.tsx?$/.test(file))
  .filter((file) => !/(^|\/)(__tests__|stories)\//.test(file))
  .sort();

describe("InputRenderer modules", () => {
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
  );
});
