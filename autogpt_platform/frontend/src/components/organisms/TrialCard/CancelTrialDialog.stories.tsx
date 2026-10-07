import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn, screen, userEvent, within } from "storybook/test";
import { CancelTrialDialog } from "./CancelTrialDialog";

const meta = {
  title: "Organisms/CancelTrialDialog",
  component: CancelTrialDialog,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "The Cancel trial button and its confirmation dialog. The dialog's open state is internal: the Open story opens it by clicking the trigger, and `defaultOpen` starts it open.",
      },
      story: { inline: false, iframeHeight: 480 },
    },
  },
  args: {
    isCanceling: false,
    onCancel: fn(),
  },
} satisfies Meta<typeof CancelTrialDialog>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Open: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Cancel trial" }),
    );
    await screen.findByRole("button", { name: "End trial now" });
  },
};

export const DefaultOpen: Story = {
  args: { defaultOpen: true },
};

export const Canceling: Story = {
  args: { isCanceling: true },
};
