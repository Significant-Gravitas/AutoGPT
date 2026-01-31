import type { Meta, StoryObj } from "@storybook/nextjs";
import { Separator } from "./Separator";

const meta: Meta<typeof Separator> = {
  title: "Atoms/Separator",
  component: Separator,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof Separator>;

export const Horizontal: Story = {
  render: () => (
    <div className="w-full max-w-md">
      <p className="mb-4 text-neutral-700 dark:text-neutral-300">
        Content above the separator
      </p>
      <Separator />
      <p className="mt-4 text-neutral-700 dark:text-neutral-300">
        Content below the separator
      </p>
    </div>
  ),
};

export const Vertical: Story = {
  render: () => (
    <div className="flex h-16 items-center gap-4">
      <span className="text-neutral-700 dark:text-neutral-300">Left</span>
      <Separator orientation="vertical" />
      <span className="text-neutral-700 dark:text-neutral-300">Right</span>
    </div>
  ),
};

export const WithCustomStyles: Story = {
  render: () => (
    <div className="w-full max-w-md space-y-4">
      <Separator className="bg-violet-500" />
      <Separator className="h-0.5 bg-gradient-to-r from-violet-500 to-blue-500" />
      <Separator className="bg-neutral-400 dark:bg-neutral-600" />
    </div>
  ),
};

export const InSection: Story = {
  render: () => (
    <div className="w-full max-w-md space-y-6">
      <section>
        <h2 className="mb-2 text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Featured Agents
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400">
          Browse our collection of featured AI agents.
        </p>
      </section>
      <Separator className="my-6" />
      <section>
        <h2 className="mb-2 text-lg font-semibold text-neutral-900 dark:text-neutral-100">
          Top Creators
        </h2>
        <p className="text-neutral-600 dark:text-neutral-400">
          Meet the creators behind the most popular agents.
        </p>
      </section>
    </div>
  ),
};
