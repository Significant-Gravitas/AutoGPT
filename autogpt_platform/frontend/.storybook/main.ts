import type { StorybookConfig } from "@storybook/nextjs";

const config: StorybookConfig = {
  stories: [
    "../src/components/overview.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/tokens/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/atoms/**/*.stories.@(js|jsx|mjs|ts|tsx)",
    "../src/components/molecules/**/*.stories.@(js|jsx|mjs|ts|tsx)",
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
};

export default config;
