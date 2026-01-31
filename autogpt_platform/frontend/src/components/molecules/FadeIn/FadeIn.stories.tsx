import type { Meta, StoryObj } from "@storybook/nextjs";
import { FadeIn } from "./FadeIn";

const meta: Meta<typeof FadeIn> = {
  title: "Molecules/FadeIn",
  component: FadeIn,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
  },
  argTypes: {
    direction: {
      control: "select",
      options: ["up", "down", "left", "right", "none"],
    },
  },
};

export default meta;
type Story = StoryObj<typeof FadeIn>;

const DemoCard = ({ title }: { title: string }) => (
  <div className="rounded-xl bg-neutral-100 p-6 dark:bg-neutral-800">
    <h3 className="mb-2 text-lg font-semibold text-neutral-900 dark:text-neutral-100">
      {title}
    </h3>
    <p className="text-neutral-600 dark:text-neutral-400">
      This card fades in with a smooth animation.
    </p>
  </div>
);

export const Default: Story = {
  args: {
    direction: "up",
    children: <DemoCard title="Fade Up" />,
  },
};

export const FadeDown: Story = {
  args: {
    direction: "down",
    children: <DemoCard title="Fade Down" />,
  },
};

export const FadeLeft: Story = {
  args: {
    direction: "left",
    children: <DemoCard title="Fade Left" />,
  },
};

export const FadeRight: Story = {
  args: {
    direction: "right",
    children: <DemoCard title="Fade Right" />,
  },
};

export const FadeOnly: Story = {
  args: {
    direction: "none",
    children: <DemoCard title="Fade Only (No Direction)" />,
  },
};

export const WithDelay: Story = {
  args: {
    direction: "up",
    delay: 0.5,
    children: <DemoCard title="Delayed Fade (0.5s)" />,
  },
};

export const SlowAnimation: Story = {
  args: {
    direction: "up",
    duration: 1.5,
    children: <DemoCard title="Slow Animation (1.5s)" />,
  },
};

export const LargeDistance: Story = {
  args: {
    direction: "up",
    distance: 60,
    children: <DemoCard title="Large Distance (60px)" />,
  },
};

export const MultipleElements: Story = {
  render: () => (
    <div className="space-y-4">
      <FadeIn direction="up" delay={0}>
        <DemoCard title="First Card" />
      </FadeIn>
      <FadeIn direction="up" delay={0.1}>
        <DemoCard title="Second Card" />
      </FadeIn>
      <FadeIn direction="up" delay={0.2}>
        <DemoCard title="Third Card" />
      </FadeIn>
    </div>
  ),
};

export const HeroExample: Story = {
  render: () => (
    <div className="text-center">
      <FadeIn direction="down" delay={0}>
        <h1 className="mb-4 text-4xl font-bold text-neutral-900 dark:text-neutral-100">
          Welcome to the Marketplace
        </h1>
      </FadeIn>
      <FadeIn direction="up" delay={0.2}>
        <p className="mb-8 text-xl text-neutral-600 dark:text-neutral-400">
          Discover AI agents built by the community
        </p>
      </FadeIn>
      <FadeIn direction="up" delay={0.4}>
        <button className="rounded-full bg-violet-600 px-8 py-3 text-white hover:bg-violet-700">
          Get Started
        </button>
      </FadeIn>
    </div>
  ),
};
