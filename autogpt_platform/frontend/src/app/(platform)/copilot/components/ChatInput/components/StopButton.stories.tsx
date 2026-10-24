import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { StopButton } from "./StopButton";

const meta: Meta<typeof StopButton> = {
  title: "Copilot/StopButton",
  component: StopButton,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Takes the send button's place while a reply streams. It arms after one second so a stray second tap on Send does not stop the reply.",
      },
    },
  },
  args: {
    onStop: () => undefined,
    // The class ChatInput gives the send slot on the default composer.
    className: "max-md:size-10",
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="pt-12">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};
