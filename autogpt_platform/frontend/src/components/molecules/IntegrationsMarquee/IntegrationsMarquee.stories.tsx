import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { IntegrationsMarquee } from "./IntegrationsMarquee";

const meta = {
  title: "Molecules/IntegrationsMarquee",
  component: IntegrationsMarquee,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "Decorative, `aria-hidden` marquee of ghost integration cards: two rows of provider logos scrolling in opposite directions behind an edge fade mask. It stays still for viewers who prefer reduced motion (read from the OS setting via `useReducedMotion`).",
      },
    },
  },
  argTypes: {
    variant: {
      control: "inline-radio",
      options: ["light", "dark"],
      description: "Card treatment for light or dark backgrounds",
    },
  },
  args: { variant: "light" },
} satisfies Meta<typeof IntegrationsMarquee>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Light: Story = {};

export const Dark: Story = {
  args: { variant: "dark" },
  decorators: [
    (Story) => (
      <div className="rounded-2xl bg-zinc-900 p-6">
        <Story />
      </div>
    ),
  ],
};

export const CustomSize: Story = {
  args: { className: "h-40 w-96" },
  parameters: {
    docs: {
      description: {
        story: "`className` overrides the default 340 x 200 frame.",
      },
    },
  },
};
