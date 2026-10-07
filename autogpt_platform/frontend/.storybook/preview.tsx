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
import { fonts } from "../src/components/styles/fonts";
import { theme } from "./theme";

// Same next/font instances as src/app/layout.tsx. The variables go on <html>
// so portalled content (dialogs, popovers, toasts) picks them up too.
document.documentElement.classList.add(
  fonts.poppins.variable,
  fonts.sans.variable,
  fonts.mono.variable,
);

// One QueryClient per story, so a story's MSW handlers are never shadowed
// by data another story cached under the same query key. Retries are off so
// failing handlers surface immediately instead of hiding behind backoff.
const storyQueryClients = new Map<string, QueryClient>();

function getStoryQueryClient(storyId: string) {
  let client = storyQueryClients.get(storyId);
  if (!client) {
    client = new QueryClient({
      defaultOptions: {
        queries: { retry: false, refetchOnWindowFocus: false },
        mutations: { retry: false },
      },
    });
    storyQueryClients.set(storyId, client);
  }
  return client;
}

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
  loaders: [mswLoader()],
  decorators: [
    (Story, context) => (
      <QueryClientProvider client={getStoryQueryClient(context.id)}>
        <div className="bg-background p-8">
          <Story />
        </div>
      </QueryClientProvider>
    ),
  ],
};

export default preview;
