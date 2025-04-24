import type { StorybookConfig } from "@storybook/nextjs";

const config: StorybookConfig = {
  stories: ["../src/**/*.mdx", "../src/**/*.stories.@(js|jsx|mjs|ts|tsx)"],
  addons: [
    "@storybook/addon-a11y",
    "@storybook/addon-onboarding",
    "@storybook/addon-links",
    "@storybook/addon-essentials",
    "@storybook/addon-interactions",
  ],
  env: {
    NEXT_PUBLIC_SUPABASE_URL: "https://your-project.supabase.co",
    NEXT_PUBLIC_SUPABASE_ANON_KEY: "your-anon-key",
  },
  features: {
    experimentalRSC: true,
  },
  framework: {
    name: "@storybook/nextjs",
    options: {},
  },
  staticDirs: [
    "../public",
    {
      from: "../node_modules/geist/dist/fonts/geist-sans",
      to: "/fonts/geist-sans",
    },
    {
      from: "../node_modules/geist/dist/fonts/geist-mono",
      to: "/fonts/geist-mono",
    },
  ],
};
export default config;
