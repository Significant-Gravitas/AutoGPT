import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// What /home downloads before it is usable is everything its layouts and page
// import statically; a dynamic import() starts a chunk that loads later. These
// libraries and panels are only needed once a message, a form, an output or a
// dialog shows, so none of them may be reachable without a dynamic import.
// Server components' imports are walked too, which errs on the strict side.
const SRC_ROOT = path.resolve(__dirname, "../../../..");
const ENTRIES = [
  "app/layout.tsx",
  "app/(platform)/layout.tsx",
  "app/(platform)/home/page.tsx",
];

const LAZY_PACKAGES = [
  // The message renderer: markdown, math and code highlighting.
  "streamdown",
  "@streamdown/math",
  "rehype-katex",
  "katex",
  // Outputs and attachments.
  "rehype-highlight",
  "highlight.js",
  // The Files tab's download-all.
  "jszip",
  // The tool cards' forms and the builder's graph library under them.
  "@rjsf/core",
  "@rjsf/utils",
  "@rjsf/validator-ajv8",
  "react-day-picker",
  "@xyflow/react",
];

const LAZY_MODULES = [
  "components/renderers/InputRenderer/FormRenderer.tsx",
  "components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView.tsx",
  "components/contextual/IntegrationsPanel/IntegrationsPanel.tsx",
  "app/(platform)/copilot/tools/RunAgent/components/AgentDetailsCard/AgentDetailsCard.tsx",
];

const STATIC_IMPORT =
  /(?:^|\n)\s*(?:(import|export)\s([^;]*?)\sfrom\s+["']([^"']+)["']|import\s+["']([^"']+)["'])/g;

const graph = walkStaticImports();

describe("/home's first download", () => {
  it("walks the page's modules", () => {
    expect(graph.parent.has("app/(platform)/copilot/CopilotPage.tsx")).toBe(
      true,
    );
  });

  it.each(LAZY_PACKAGES)("leaves %s to a later chunk", (pkg) => {
    const importer = graph.packageImporters.get(pkg);
    expect(importer && chainTo(importer), `${pkg} is imported`).toBeFalsy();
  });

  it.each(LAZY_MODULES)("leaves %s to a later chunk", (module) => {
    expect(
      graph.parent.has(module) && chainTo(module),
      `${module} is imported`,
    ).toBeFalsy();
  });
});

function walkStaticImports() {
  const parent = new Map<string, string | null>(
    ENTRIES.map((entry) => [entry, null]),
  );
  const packageImporters = new Map<string, string>();
  const queue = [...ENTRIES];
  while (queue.length > 0) {
    const file = queue.shift()!;
    for (const specifier of staticImportsOf(file)) {
      const target = resolve(file, specifier);
      if (target === null) {
        const pkg = packageName(specifier);
        if (!packageImporters.has(pkg)) packageImporters.set(pkg, file);
      } else if (target && !parent.has(target)) {
        parent.set(target, file);
        queue.push(target);
      }
    }
  }
  return { parent, packageImporters };
}

/** Specifiers `file` imports at run time; type-only imports are erased. */
function staticImportsOf(file: string): string[] {
  const source = readFileSync(path.join(SRC_ROOT, file), "utf8").replace(
    /\/\*[\s\S]*?\*\//g,
    "",
  );
  const found: string[] = [];
  for (const [, keyword, clause, specifier, bare] of source.matchAll(
    STATIC_IMPORT,
  )) {
    if (bare) {
      found.push(bare);
      continue;
    }
    if (/^type\s/.test(clause)) continue;
    const named = clause.match(/^\{([\s\S]*)\}$/);
    if (keyword === "import" && named && isAllTypes(named[1])) continue;
    found.push(specifier);
  }
  return found;
}

function isAllTypes(names: string) {
  const list = names
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  return list.length > 0 && list.every((name) => name.startsWith("type "));
}

/** A path under SRC_ROOT, "" for a file that isn't code, or null for a package. */
function resolve(from: string, specifier: string): string | null {
  let target: string;
  if (specifier.startsWith("@/")) {
    target = specifier.slice(2);
  } else if (specifier.startsWith(".")) {
    target = path.posix.normalize(
      path.posix.join(path.posix.dirname(from), specifier),
    );
  } else {
    return null;
  }
  const candidates = [
    `${target}.ts`,
    `${target}.tsx`,
    `${target}/index.ts`,
    `${target}/index.tsx`,
    target,
  ];
  const found = candidates.find((candidate) =>
    existsSync(path.join(SRC_ROOT, candidate)),
  );
  return found && /\.tsx?$/.test(found) ? found : "";
}

function packageName(specifier: string) {
  const parts = specifier.split("/");
  return specifier.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
}

function chainTo(module: string) {
  const chain: string[] = [];
  for (
    let current: string | null | undefined = module;
    current;
    current = graph.parent.get(current)
  ) {
    chain.unshift(current);
  }
  return chain.join("\n  -> ");
}
