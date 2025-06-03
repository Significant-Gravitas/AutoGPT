import React from "react";
import type { Preview } from "@storybook/react";
import { initialize, mswLoader } from "msw-storybook-addon";
import "../src/app/globals.css";

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
      <div
        className="font-sans"
        style={
          {
            fontFamily: "Poppins, system-ui, -apple-system, sans-serif",
            "--font-poppins": "Poppins, sans-serif",
            "--font-geist-sans": "system-ui, -apple-system, sans-serif",
            "--font-geist-mono": "ui-monospace, monospace",
          } as React.CSSProperties
        }
      >
        <Story />
      </div>
    ),
  ],
};

export default preview;
