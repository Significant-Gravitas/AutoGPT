import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import validator from "@rjsf/validator-ajv8";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";

// The renderer must load whichever of its modules loads first. The production
// server's start-up preload loads modules in module-id order, so whichever
// module that happens to be decides it, and a dependency change can reorder it.
// A module loaded in the middle of an import cycle runs before some of its
// imports have initialised, and reading one then throws "Cannot access '<x>'
// before initialization". So registry/Form.tsx, and the registries it builds
// the form's theme from, read no field component until the form first renders.
const SRC_ROOT = path.resolve(__dirname, "../../../..");
const RENDERER_ROOT = path.resolve(__dirname, "..");
const FORM = "registry/Form.tsx";

// Builder modules on import chains from the renderer back to Form: fields read
// the node and edge stores, the stores name CustomNode, CustomNode renders
// FormCreator, and FormCreator renders FormRenderer. The stores name
// CustomNode only as a type today, which compiles away, so no chain closes yet.
const BUILDER_MODULES = [
  "app/(platform)/build/stores/nodeStore.ts",
  "app/(platform)/build/stores/edgeStore.ts",
  "app/(platform)/build/stores/historyStore.ts",
  "app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode.tsx",
  "app/(platform)/build/components/FlowEditor/nodes/FormCreator.tsx",
];

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

/** A stand-in for a module's exports that records each read of one. */
function watchReads(
  file: string,
  exports: Record<string, unknown>,
  reads: string[],
) {
  const watched: Record<string, unknown> = {};
  for (const name of Object.keys(exports)) {
    Object.defineProperty(watched, name, {
      enumerable: true,
      get() {
        reads.push(`${file}: ${name}`);
        return exports[name];
      },
    });
  }
  return watched;
}

describe("InputRenderer modules", () => {
  it("finds the renderer's modules", () => {
    expect(modules).toContain(FORM);
  });

  it("nothing Form imports leads back to Form", () => {
    expect(
      cycleThroughForm(),
      "on an import cycle, a module can load before what it imports",
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

  it.each(BUILDER_MODULES)(
    "builder module %s loads when it is the first module imported",
    async (file) => {
      vi.resetModules();
      await expect(import(path.join(SRC_ROOT, file))).resolves.toBeDefined();
    },
    CASE_TIMEOUT_MS,
  );

  // Last: it loads the renderer, and loading it before the cases above could
  // hide a cycle from them.
  it("loading Form reads no field module until the form first renders", async () => {
    const reads: string[] = [];
    // What the registries list the fields, templates and widgets from.
    const fieldModules = [...new Set(importsOf(FORM).flatMap(importsOf))];
    vi.resetModules();
    for (const file of fieldModules) {
      const id = path.join(RENDERER_ROOT, file);
      vi.doMock(id, async () =>
        watchReads(
          file,
          await vi.importActual<Record<string, unknown>>(id),
          reads,
        ),
      );
    }
    try {
      const { default: Form } = (await import(
        path.join(RENDERER_ROOT, FORM)
      )) as typeof import("../registry/Form");
      expect(reads, "what loading Form read").toEqual([]);

      const { render } = await import("@/tests/integrations/test-utils");
      const form = createElement(Form, {
        schema: { type: "string" },
        validator,
        formContext: { showHandles: false },
      });
      render(form);
      render(form);
      // Only the theme reads NumberField, and the theme is built once.
      expect(
        reads.filter(
          (read) => read === "base/number/NumberField.tsx: NumberField",
        ),
      ).toHaveLength(1);
    } finally {
      for (const file of fieldModules) {
        vi.doUnmock(path.join(RENDERER_ROOT, file));
      }
      vi.resetModules();
    }
  });
});
