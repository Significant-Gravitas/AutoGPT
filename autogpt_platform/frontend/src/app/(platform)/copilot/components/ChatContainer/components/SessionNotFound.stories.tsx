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

export const RemovedByReset: StoryObj<typeof SessionNotFound> = {
  args: { removedChat: { expertId: "expert-1", expertName: "Remy" } },
};

export const RemovedByResetExpertGone: StoryObj<typeof SessionNotFound> = {
  args: { removedChat: { expertId: null, expertName: "Remy" } },
};
