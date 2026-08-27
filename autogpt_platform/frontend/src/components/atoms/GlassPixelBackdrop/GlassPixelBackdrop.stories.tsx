import type { Meta, StoryObj } from "@storybook/nextjs";
import { GlassPixelBackdrop } from "./GlassPixelBackdrop";

const meta: Meta<typeof GlassPixelBackdrop> = {
  title: "Atoms/GlassPixelBackdrop",
  component: GlassPixelBackdrop,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A glass pixel mosaic backdrop: a grid of frosted white tiles at deterministic varying opacities, letting the gradient behind it glow through each pixel differently. Positioned absolutely, so it needs a relatively-positioned parent.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory() {
    return (
      <div className="relative h-64 w-96 overflow-hidden rounded-2xl bg-gradient-to-br from-violet-500 via-purple-500 to-indigo-600">
        <GlassPixelBackdrop />
      </div>
    );
  },
};
