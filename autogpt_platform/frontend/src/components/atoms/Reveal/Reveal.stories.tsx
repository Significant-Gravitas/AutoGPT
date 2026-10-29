import type { Meta, StoryObj } from "@storybook/nextjs";
import { RevealGroup, RevealItem } from "./Reveal";

const meta: Meta<typeof RevealGroup> = {
  title: "Atoms/Reveal",
  component: RevealGroup,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Entrance reveal for stacked content: each RevealItem blurs and lifts into place independently, so items mounted later still animate. Honors prefers-reduced-motion by falling back to a plain fade.",
      },
    },
  },
  argTypes: {
    className: { control: "text", description: "Optional className." },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory(args) {
    return (
      <RevealGroup {...args} className="flex w-72 flex-col gap-3">
        <RevealItem>
          <h3 className="text-lg font-semibold text-zinc-900">
            Tell us about your work
          </h3>
        </RevealItem>
        <RevealItem>
          <p className="text-sm text-zinc-600">
            Most people talk for two to three minutes.
          </p>
        </RevealItem>
        <RevealItem>
          <p className="text-sm text-zinc-600">
            Each item reveals on its own, even if it mounts later.
          </p>
        </RevealItem>
      </RevealGroup>
    );
  },
};
