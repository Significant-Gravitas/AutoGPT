import React from "react";
import type { Preview } from "@storybook/react";
import { initialize, mswLoader } from "msw-storybook-addon";
import "../src/app/globals.css";
import "../src/components/styles/fonts.css";

// Initialize MSW
initialize();

const preview: Preview = {
  parameters: {
    nextjs: {
      appDirectory: true,
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  loaders: [mswLoader],
  decorators: [
    (Story) => (
      <>
        <Story />
      </>
    ),
  ],
};

export default preview;
