import type { Meta, StoryObj } from "@storybook/nextjs";
import { Emoji } from "./Emoji";

const meta: Meta<typeof Emoji> = {
  title: "Atoms/Emoji",
  tags: ["autodocs"],
  component: Emoji,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Renders emoji text as cross-platform SVG images using Twemoji.",
      },
    },
  },
  argTypes: {
    text: {
      control: "text",
      description: "Emoji character(s) to render",
    },
    size: {
      control: { type: "number", min: 12, max: 96, step: 4 },
      description: "Size in pixels (width and height)",
    },
  },
  args: {
    text: "🚀",
    size: 24,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    text: "🚀",
  },
};

export const Small: Story = {
  args: {
    text: "✨",
    size: 16,
  },
};

export const Large: Story = {
  args: {
    text: "🎉",
    size: 48,
  },
};

export const AllSizes: Story = {
  render: renderAllSizes,
};

function renderAllSizes() {
  const sizes = [16, 24, 32, 48, 64];
  return (
    <div className="flex items-end gap-4">
      {sizes.map((size) => (
        <div key={size} className="flex flex-col items-center gap-2">
          <Emoji text="🔥" size={size} />
          <span className="text-xs text-muted-foreground">{size}px</span>
        </div>
      ))}
    </div>
  );
}

export const MultipleEmojis: Story = {
  render: renderMultipleEmojis,
};

function renderMultipleEmojis() {
  const emojis = ["😀", "🎯", "⚡", "🌈", "🤖", "💡", "🔧", "📦"];
  return (
    <div className="flex flex-wrap gap-3">
      {emojis.map((emoji) => (
        <Emoji key={emoji} text={emoji} size={32} />
      ))}
    </div>
  );
}
