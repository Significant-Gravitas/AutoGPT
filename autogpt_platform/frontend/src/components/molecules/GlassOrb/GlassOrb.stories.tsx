import type { Meta, StoryObj } from "@storybook/nextjs";
import { GlassOrb } from "./GlassOrb";

const meta: Meta<typeof GlassOrb> = {
  title: "Molecules/GlassOrb",
  component: GlassOrb,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A glass orb: distorted gradient blobs drifting over each other, sealed under an Aave-style frosted glass pane with refraction. Fills its parent, so it needs a relatively-positioned sized container.",
      },
    },
  },
  argTypes: {
    params: {
      control: "object",
      description:
        "Glass tuning knobs: frost, saturation, tint, edge, distortion, ringWidth, ringDepth, ringDark.",
    },
  },
  args: {
    params: {
      frost: 1.5,
      saturation: 1.5,
      tint: 0.12,
      edge: 0.55,
      distortion: 8,
      ringWidth: 1,
      ringDepth: 2,
      ringDark: 0.25,
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory(args) {
    return (
      <span className="relative block size-24">
        <GlassOrb {...args} />
      </span>
    );
  },
};
