import {
  Controls,
  Primary,
  Source,
  Stories,
  Subtitle,
  Title,
} from "@storybook/addon-docs/blocks";
import { Preview } from "@storybook/nextjs";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { mswLoader } from "msw-storybook-addon/csf3";
import React from "react";
import "../src/app/globals.css";
import "../src/app/tokens/themes/canary.css";
import "../src/components/styles/fonts.css";
import { theme } from "./theme";

// One QueryClient per browser session is fine for Storybook — retries
// are off so failing MSW handlers surface immediately instead of being
// hidden behind exponential backoff.
const storyQueryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});

// The toolbar's "Look": canary gives every token a loud value, so whatever
// keeps its normal colour, corner or font is not reading a token.
function applyLook(look: unknown) {
  if (look === "canary") document.documentElement.dataset.look = "canary";
  else delete document.documentElement.dataset.look;
}

const preview: Preview = {
  globalTypes: {
    look: {
      description: "Token look",
      toolbar: {
        title: "Look",
        icon: "paintbrush",
        items: [
          { value: "default", title: "Default" },
          { value: "canary", title: "Canary (shows what bypasses the tokens)" },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: { look: "default" },
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
  loaders: [mswLoader()],
  decorators: [
    (Story, context) => {
      applyLook(context.globals.look);
      return (
        <QueryClientProvider client={storyQueryClient}>
          <div className="bg-canvas p-8">
            <Story />
          </div>
        </QueryClientProvider>
      );
    },
  ],
};

export default preview;
