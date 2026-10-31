import type { Meta, StoryObj } from "@storybook/nextjs";
import { NuqsAdapter } from "nuqs/adapters/react";
import { SessionNotFound } from "./SessionNotFound";

const meta: Meta<typeof SessionNotFound> = {
  title: "Copilot/SessionNotFound",
  component: SessionNotFound,
  decorators: [
    (Story) => (
      <NuqsAdapter>
        <div style={{ height: "100vh" }}>
          <Story />
        </div>
      </NuqsAdapter>
    ),
  ],
  parameters: { layout: "fullscreen" },
};
export default meta;

export const Default: StoryObj<typeof SessionNotFound> = {};
