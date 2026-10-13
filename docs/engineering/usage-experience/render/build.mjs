import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { composerPlugin } from "./composer-plugin.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../../..");
const frontend = path.join(repo, "autogpt_platform/frontend");
const require = createRequire(path.join(frontend, "package.json"));
const viteRequire = createRequire(
  createRequire(require.resolve("vitest/package.json")).resolve(
    "vite/package.json",
  ),
);
const esbuild = viteRequire("esbuild");
const postcss = require("postcss");
const tailwind = require("tailwindcss");
const output = process.env.USAGE_RENDER_OUTPUT || "/tmp/autogpt-usage-render";
await fs.mkdir(output, { recursive: true });
const mocks = path.join(here, "mocks.tsx");
const result = await esbuild.build({
  entryPoints: [path.join(here, "main.tsx")],
  outfile: path.join(output, "bundle.js"),
  bundle: true,
  minify: true,
  format: "iife",
  jsx: "automatic",
  platform: "browser",
  nodePaths: [path.join(frontend, "node_modules")],
  plugins: [
    composerPlugin,
    {
      name: "fixture-services",
      setup(build) {
        build.onResolve(
          {
            filter:
              /(^@sentry\/nextjs$|^next\/(link|navigation)$|\/(useUsageExperience|useProActivation|useWorkspaceStorage|useTrialStatus|useTrialCheckoutReturn|useTrialCard|usePaymentMethodCard|useInvoicesCard|useYourPlanCard|usePaywallModal|AutomationCreditsTab|monetization-analytics)$|__generated__\/endpoints\/(credits\/credits|trials\/trials|chat\/chat)$|lib\/autogpt-server-api\/helpers$)/,
          },
          () => ({ path: mocks }),
        );
      },
    },
  ],
  define: {
    "process.env.NODE_ENV": '"production"',
    "process.env.NEXT_PUBLIC_BEHAVE_AS": '"CLOUD"',
    "process.env.NEXT_PUBLIC_APP_ENV": '"prod"',
    "process.env": "{}",
  },
  alias: {
    "@": path.join(frontend, "src"),
    react: path.dirname(require.resolve("react/package.json")),
    "react-dom": path.dirname(require.resolve("react-dom/package.json")),
  },
  metafile: true,
});
const tailwindResult = await esbuild.build({
  metafile: true,
  entryPoints: [path.join(frontend, "tailwind.config.ts")],
  outfile: path.join(output, "tailwind.cjs"),
  bundle: true,
  platform: "node",
  format: "cjs",
  packages: "external",
  nodePaths: [path.join(frontend, "node_modules")],
});
// Resolve Tailwind's configuration dependencies through the app installation.
let configSource = await fs.readFile(path.join(output, "tailwind.cjs"), "utf8");
configSource = configSource.replace(
  /require\("([^".][^"]*)"\)/g,
  (_, specifier) => `require(${JSON.stringify(require.resolve(specifier))})`,
);
await fs.writeFile(path.join(output, "tailwind.cjs"), configSource);
const config = require(path.join(output, "tailwind.cjs")).default;
config.content = [
  {
    raw: await fs.readFile(path.join(output, "bundle.js"), "utf8"),
    extension: "js",
  },
];
const compiled = await postcss([tailwind(config)]).process(
  await fs.readFile(path.join(frontend, "src/app/globals.css"), "utf8"),
  { from: path.join(frontend, "src/app/globals.css") },
);
const fontCSS = await fs.readFile(path.join(here, "fonts.css"), "utf8");
await fs.cp(path.join(here, "fonts"), path.join(output, "fonts"), {
  recursive: true,
});
await fs.writeFile(
  path.join(output, "style.css"),
  fontCSS +
    '\n:root{--font-poppins:Poppins,"Poppins Fallback";--font-geist-sans:GeistSans,"GeistSans Fallback";--font-geist-mono:monospace}\n' +
    compiled.css,
);
await fs.writeFile(
  path.join(output, "index.html"),
  '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="/style.css"><title>AutoGPT usage component screenshots</title></head><body><div id="autogpt-system-redesign"></div><script src="/bundle.js"></script></body></html>',
);
const fixtureFiles = (await fs.readdir(here))
  .filter((name) => /\.(tsx?|mjs|css)$/.test(name))
  .map((name) => path.join(here, name));
const fonts = (await fs.readdir(path.join(here, "fonts"))).map((name) =>
  path.join(here, "fonts", name),
);
const inputs = [
  ...new Set(
    [
      ...[
        ...Object.keys(result.metafile.inputs),
        ...Object.keys(tailwindResult.metafile.inputs),
      ].filter(
        (file) => !file.startsWith("<") && path.resolve(file).startsWith(repo),
      ),
      path.join(frontend, "src/app/globals.css"),
      path.join(frontend, "tailwind.config.ts"),
      path.join(frontend, "package.json"),
      path.join(frontend, "pnpm-lock.yaml"),
      path.join(frontend, "src/components/styles/fonts.ts"),
      ...fixtureFiles,
      ...fonts,
    ].map((file) => path.resolve(file)),
  ),
];
const sourceFiles = [];
for (const file of inputs.sort())
  sourceFiles.push({
    path: path.relative(repo, path.resolve(file)),
    sha256: crypto
      .createHash("sha256")
      .update(await fs.readFile(file))
      .digest("hex"),
  });
await fs.writeFile(
  path.join(output, "source-manifest.json"),
  JSON.stringify({ sourceFiles }, null, 2),
);
console.log(JSON.stringify({ output, sourceFiles: sourceFiles.length }));
