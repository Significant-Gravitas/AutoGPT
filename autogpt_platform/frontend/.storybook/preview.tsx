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
import { initialize, mswLoader } from "msw-storybook-addon";
import React from "react";
import "../src/app/globals.css";
import "../src/components/styles/fonts.css";
import { theme } from "./theme";

// Initialize MSW
initialize();

// One QueryClient per browser session is fine for Storybook — retries
// are off so failing MSW handlers surface immediately instead of being
// hidden behind exponential backoff.
const storyQueryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false },
    mutations: { retry: false },
  },
});

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
      <QueryClientProvider client={storyQueryClient}>
        <div className="bg-background p-8">
          <Story />
        </div>
      </QueryClientProvider>
    ),
  ],
};

export default preview;
