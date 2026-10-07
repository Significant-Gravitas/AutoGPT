import type { StorybookConfig } from "@storybook/nextjs";
import fs from "node:fs";
import path from "node:path";
import webpack from "webpack";

// Packages that ship next/font calls and must go through Next's SWC
// transform, mirroring transpilePackages in next.config.mjs.
const NEXT_FONT_PACKAGES = ["geist"];

// Storybook excludes node_modules from its SWC rule except for
// transpilePackages, but its regex only allows packages that sit directly
// under node_modules. pnpm nests them under node_modules/.pnpm/<id>/, so
// geist's next/font/local call reached the browser untransformed.
const pnpmAwareNodeModulesExclude = new RegExp(
  `node_modules/(?!(\\.pnpm/[^/]+/node_modules/)?(${NEXT_FONT_PACKAGES.join("|")})/)`,
);

// Storybook's next/font/local shim points @font-face at the font file's path
// relative to the project root, which is only reachable if that folder is
// served. Serve geist's fonts at the same (pnpm) path.
const frontendRoot = path.resolve(__dirname, "..");
const geistFontsDir = path.join(
  fs.realpathSync(path.join(frontendRoot, "node_modules/geist")),
  "dist/fonts",
);

interface NextSwcRule {
  use: { loader: string };
  exclude: unknown[];
}

function isNextSwcRule(rule: unknown): rule is NextSwcRule {
  if (!rule || typeof rule !== "object") return false;
  if (!("use" in rule) || !("exclude" in rule)) return false;
  const { use, exclude } = rule;
  return (
    Array.isArray(exclude) &&
    !!use &&
    typeof use === "object" &&
    "loader" in use &&
    typeof use.loader === "string" &&
    use.loader.includes("next-swc-loader-patch")
  );
}

const config: StorybookConfig = {
  stories: [
    "../src/components/overview.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/tokens/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/atoms/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/molecules/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/organisms/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/ai-elements/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/renderers/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/layout/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/contextual/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/app/[(]platform[)]/copilot/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/app/[(]platform[)]/components/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/app/[(]platform[)]/marketplace/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/app/[(]platform[)]/artifacts/**/*.stories.@(js|jsx|mjs|ts|tsx)",
  ],
  addons: [
    "@storybook/addon-a11y",
    "@storybook/addon-onboarding",
    "@storybook/addon-links",
    "@storybook/addon-docs",
    "msw-storybook-addon",
  ],
  features: {
    experimentalRSC: true,
  },
  framework: {
    name: "@storybook/nextjs",
    options: { builder: { useSWC: true } },
  },
  staticDirs: [
    "../public",
    {
      from: geistFontsDir,
      to: path.relative(frontendRoot, geistFontsDir),
    },
  ],
  webpackFinal: async (config) => {
    // Client code reaches the auth server actions (API client, avatar
    // upload), and importing that module drags the database client into the
    // preview bundle. Stories get a signed-out stand-in instead.
    config.plugins ??= [];
    config.plugins.push(
      new webpack.NormalModuleReplacementPlugin(
        /(^|[\\/])lib[\\/]auth[\\/]actions(\.ts)?$/,
        path.resolve(__dirname, "mocks/auth-actions.ts"),
      ),
    );
    for (const rule of config.module?.rules ?? []) {
      if (isNextSwcRule(rule)) {
        rule.exclude = [pnpmAwareNodeModulesExclude, ...rule.exclude.slice(1)];
      }
    }
    return config;
  },
};

export default config;
