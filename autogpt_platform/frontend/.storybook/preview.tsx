import {
  Controls,
  Primary,
  Source,
  Stories,
  Subtitle,
  Title,
} from "@storybook/addon-docs/blocks";
import { Preview } from "@storybook/nextjs";
import { initialize, mswLoader } from "msw-storybook-addon";
import React from "react";
import "../src/app/globals.css";
import "../src/components/styles/fonts.css";
import { theme } from "./theme";

// Initialize MSW
initialize();

const preview: Preview = {
  parameters: {
    nextjs: {
      appDirectory: true,
    },
    docs: {
      theme,
      page: () => (
        <>
          <Title />
          <Subtitle />

          <Primary />
          <Source />
          <Stories />
          <Controls />
        </>
      ),
    },
  },
  loaders: [mswLoader],
  decorators: [
    (Story) => (
      <div className="bg-background p-8">
        <Story />
      </div>
    ),
  ],
};

export default preview;
