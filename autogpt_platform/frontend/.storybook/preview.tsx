import {
  Controls,
  Primary,
  Source,
  Stories,
  Subtitle,
  Title,
} from "@storybook/addon-docs/blocks";
import { Preview } from "@storybook/nextjs-vite";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { mswLoader } from "msw-storybook-addon/csf3";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import React from "react";
import "../src/app/globals.css";
import { fonts } from "../src/components/styles/fonts";
import { theme } from "./theme";

// Same next/font instances as src/app/layout.tsx. Storybook's next/font
// shim builds the `variable` class name from the weight, so Geist's
// "100 900" yields a class with a space in it and a selector that never
// matches. Set the variables from each font's family instead, on <html> so
// portalled content (dialogs, popovers, toasts) picks them up too.
const rootStyle = document.documentElement.style;
rootStyle.setProperty("--font-poppins", fonts.poppins.style.fontFamily);
rootStyle.setProperty("--font-geist-sans", fonts.sans.style.fontFamily);
rootStyle.setProperty("--font-geist-mono", fonts.mono.style.fontFamily);

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
        {/* Components that keep state in the URL (useQueryState) need an
            adapter; stories get an in-memory one. */}
        <NuqsTestingAdapter>
          <div className="bg-background p-8">
            <Story />
          </div>
        </NuqsTestingAdapter>
      </QueryClientProvider>
    ),
  ],
};

export default preview;
