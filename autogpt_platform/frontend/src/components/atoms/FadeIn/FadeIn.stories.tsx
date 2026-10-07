import { FadeIn } from "./FadeIn";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";

const meta: Meta<typeof FadeIn> = {
  title: "Atoms/FadeIn",
  tags: ["autodocs"],
  component: FadeIn,
  parameters: {
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    layout: "centered",
    docs: {
      description: {
        component:
          "A wrapper that fades in its children with a subtle upward slide animation using framer-motion.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <FadeIn>
      <div className="rounded-lg border border-zinc-200 bg-white p-8 text-center">
        <p className="text-lg font-medium">This content fades in</p>
        <p className="text-sm text-muted-foreground">
          With a subtle upward slide animation
        </p>
      </div>
    </FadeIn>
  ),
};
