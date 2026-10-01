import type { StorybookConfig } from "@storybook/nextjs";
import path from "node:path";
import webpack from "webpack";

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
    "../src/app/[(]platform[)]/marketplace/**/*.stories.@(js|jsx|mjs|ts|tsx)",
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
    return config;
  },
};

export default config;
