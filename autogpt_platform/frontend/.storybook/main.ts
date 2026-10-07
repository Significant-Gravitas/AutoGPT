import type { StorybookConfig } from "@storybook/nextjs-vite";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const storybookDir = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(storybookDir, "..");

// geist calls next/font/local from inside node_modules, which the Next.js
// Vite plugin leaves untransformed. Stories import a stand-in
// (mocks/geist.ts) that loads the same font files, served from here.
const geistFontsDir = path.join(
  fs.realpathSync(path.join(frontendRoot, "node_modules/geist")),
  "dist/fonts",
);

// Client code reaches the auth server actions (API client, avatar upload),
// and importing that module drags the database client into the preview
// bundle. Stories get a signed-out stand-in instead, whatever the import
// specifier.
function mockAuthActions(): Plugin {
  const mock = path.join(storybookDir, "mocks/auth-actions.ts");
  return {
    name: "autogpt:mock-auth-actions",
    enforce: "pre",
    async resolveId(source, importer, options) {
      const resolved = await this.resolve(source, importer, {
        ...options,
        skipSelf: true,
      });
      if (
        resolved &&
        /[\\/]src[\\/]lib[\\/]auth[\\/]actions\.ts$/.test(resolved.id)
      ) {
        return mock;
      }
      return null;
    },
  };
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
    "@storybook/addon-vitest",
    "msw-storybook-addon",
  ],
  features: {
    experimentalRSC: true,
  },
  framework: {
    name: "@storybook/nextjs-vite",
    options: {},
  },
  staticDirs: ["../public", { from: geistFontsDir, to: "/geist-fonts" }],
  viteFinal: async (config) => {
    config.plugins = [...(config.plugins ?? []), mockAuthActions()];
    config.resolve ??= {};
    config.resolve.alias = [
      ...(Array.isArray(config.resolve.alias)
        ? config.resolve.alias
        : Object.entries(config.resolve.alias ?? {}).map(
            ([find, replacement]) => ({ find, replacement }),
          )),
      {
        find: /^geist\/font\/(sans|mono)$/,
        replacement: path.join(storybookDir, "mocks/geist.ts"),
      },
    ];
    return config;
  },
};

export default config;
