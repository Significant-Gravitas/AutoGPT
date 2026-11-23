import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";

// registry/Form.tsx builds the form when it loads (`export default
// generateForm()`), reading every field component. If loading any other
// renderer module first can lead back to Form before a field has initialised,
// Form throws "Cannot access '<field>' before initialization". The production
// server's start-up preload loads modules in module-id order, so whichever
// module that happens to be decides it, and a dependency change can reorder it.
const RENDERER_ROOT = path.resolve(__dirname, "..");
const FORM = "registry/Form.tsx";

// The first case compiles the renderer graph from a cold cache, which a slow
// runner can take long over. No warm-up: one that runs the modules hides the
// cycle in some runs.
const CASE_TIMEOUT_MS = 120_000;

const modules = readdirSync(RENDERER_ROOT, {
  recursive: true,
  encoding: "utf8",
})
  .map((file) => file.split(path.sep).join("/"))
  .filter((file) => /\.tsx?$/.test(file))
  .filter((file) => !/\.(test|stories)\.tsx?$/.test(file))
  .filter((file) => !/(^|\/)(__tests__|stories)\//.test(file))
  .sort();

const IMPORT =
  /(?:^|\n)\s*((?:import|export)\s[^;]*?)\sfrom\s+["']([^"']+)["']/g;
const ALIAS = "@/components/renderers/InputRenderer/";

/** The renderer modules `file` imports at run time, as paths under
 * RENDERER_ROOT. Type-only imports are erased, so they can't form a cycle. */
function importsOf(file: string): string[] {
  const source = readFileSync(path.join(RENDERER_ROOT, file), "utf8");
  const found: string[] = [];
  for (const [, clause, specifier] of source.matchAll(IMPORT)) {
    if (/^(?:import|export)\s+type\s/.test(clause)) continue;
    let target: string;
    if (specifier.startsWith(ALIAS)) {
      target = specifier.slice(ALIAS.length);
    } else if (specifier.startsWith(".")) {
      target = path.posix.normalize(
        path.posix.join(path.posix.dirname(file), specifier),
      );
    } else {
      continue;
    }
    const resolved = [
      `${target}.ts`,
      `${target}.tsx`,
      `${target}/index.ts`,
      `${target}/index.tsx`,
    ].find((candidate) => existsSync(path.join(RENDERER_ROOT, candidate)));
    if (resolved) found.push(resolved);
  }
  return found;
}

/** A chain of imports from Form back to Form, or null when there is none. */
function cycleThroughForm(): string[] | null {
  const seen = new Set<string>();
  const walk = (file: string, chain: string[]): string[] | null => {
    for (const next of importsOf(file)) {
      if (next === FORM) return [...chain, next];
      if (seen.has(next)) continue;
      seen.add(next);
      const found = walk(next, [...chain, next]);
      if (found) return found;
    }
    return null;
  };
  return walk(FORM, [FORM]);
}

describe("InputRenderer modules", () => {
  it("finds the renderer's modules", () => {
    expect(modules).toContain(FORM);
  });

  it("nothing Form imports leads back to Form", () => {
    expect(
      cycleThroughForm(),
      "Form runs at load, so a module it reaches must not import it back",
    ).toBeNull();
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
