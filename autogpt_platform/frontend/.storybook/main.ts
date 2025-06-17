import type { StorybookConfig } from "@storybook/nextjs";

const config: StorybookConfig = {
  stories: ["../src/**/*.mdx", "../src/**/*.stories.@(js|jsx|mjs|ts|tsx)"],
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
