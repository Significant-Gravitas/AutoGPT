import type { Meta, StoryObj } from "@storybook/nextjs";
import { userEvent, within } from "storybook/test";
import { FeedbackModal } from "./FeedbackModal";

const meta: Meta<typeof FeedbackModal> = {
  title: "Copilot/FeedbackModal",
  component: FeedbackModal,
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Dialog asking what could have been better after a reply is downvoted.",
      },
    },
  },
  args: {
    isOpen: true,
    onSubmit: () => undefined,
    onCancel: () => undefined,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  name: "Open, empty",
};

export const WithComment: Story = {
  name: "Open, with a typed comment",
  play: async ({ canvasElement }) => {
    const page = within(canvasElement.ownerDocument.body);
    const field = await page.findByPlaceholderText(
      "Tell us what went wrong or could be improved...",
    );
    await userEvent.type(
      field,
      "The summary skipped the pricing section I asked about.",
    );
  },
};
