import type { Meta, StoryObj } from "@storybook/nextjs";
import { CompactionCard } from "./CompactionCard";

const meta: Meta<typeof CompactionCard> = {
  title: "Copilot/CompactionCard",
  component: CompactionCard,
  parameters: {
    docs: {
      description: {
        component:
          "Context compaction as one continuous moment. The bar approaches each phase's ceiling exponentially so it can never finish before the work does; entering the rebuild phase raises the ceiling, and the row settles into a one-line receipt when the stream moves on.",
      },
    },
  },
};
export default meta;

type Story = StoryObj<typeof CompactionCard>;

export const Summarizing: Story = {
  args: {
    phase: "summarizing",
    stats: { tokensBefore: 128_000 },
    isSettled: false,
  },
};

export const Rebuilding: Story = {
  args: {
    phase: "rebuilding",
    stats: { tokensBefore: 128_000, tokensAfter: 31_000, messagesBefore: 412 },
    isSettled: false,
  },
};

export const Settled: Story = {
  args: {
    phase: null,
    stats: {
      tokensBefore: 128_000,
      tokensAfter: 31_000,
      messagesBefore: 412,
      messagesAfter: 38,
    },
    isSettled: true,
  },
};

export const SettledLegacyRow: Story = {
  args: { phase: null, stats: {}, isSettled: true },
};

export const SettledDropped: Story = {
  args: {
    phase: null,
    stats: { dropped: true, messagesBefore: 412 },
    isSettled: true,
  },
};
