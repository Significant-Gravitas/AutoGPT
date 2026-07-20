import type { StorybookConfig } from "@storybook/nextjs";
import { createRequire } from "node:module";
import path from "node:path";

// `@autogpt/icons` is an optional (private) dependency. next.config.mjs
// aliases it to a local stub when it isn't installed, but @storybook/nextjs
// never runs next.config's custom `webpack` function, so mirror the alias here
// or `pnpm storybook` breaks for anyone without access to the package.
// Storybook loads this file through esbuild-register (CJS), where
// `import.meta.url` is shimmed to undefined — use `__filename` instead.
const require = createRequire(__filename);
const AUTOGPT_ICONS_STUB_PATH = path.resolve(
  __dirname,
  "../src/components/atoms/Icon/agptIconsStub.ts",
);
let hasAutoGPTIcons = true;
try {
  require.resolve("@autogpt/icons/package.json");
} catch {
  hasAutoGPTIcons = false;
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
    "../src/app/[(]platform[)]/copilot/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/app/[(]platform[)]/components/**/*.stories.@(js|jsx|mjs|ts|tsx)",
  ],
  addons: [
    "@storybook/addon-a11y",
    "@storybook/addon-onboarding",
    "@storybook/addon-links",
    "@storybook/addon-docs",
  ],
  features: {
    experimentalRSC: true,
  },
  framework: {
    name: "@storybook/nextjs",
    options: { builder: { useSWC: true } },
  },
  staticDirs: ["../public"],
  webpackFinal: async (webpackConfig) => {
    if (!hasAutoGPTIcons) {
      webpackConfig.resolve = webpackConfig.resolve ?? {};
      // `$` = exact match, so the /solid subpath isn't rewritten by the bare key.
      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
        "@autogpt/icons$": AUTOGPT_ICONS_STUB_PATH,
        "@autogpt/icons/solid$": AUTOGPT_ICONS_STUB_PATH,
      };
    }
    return webpackConfig;
  },
};

export default config;
