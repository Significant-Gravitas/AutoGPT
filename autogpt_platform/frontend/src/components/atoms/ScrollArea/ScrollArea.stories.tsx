import type { Meta, StoryObj } from "@storybook/nextjs";
import { Text } from "../Text/Text";
import { ScrollArea } from "./ScrollArea";

const meta: Meta<typeof ScrollArea> = {
  title: "Atoms/ScrollArea",
  component: ScrollArea,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          "Radix scroll area with a thin zinc scrollbar that matches `scrollbarStyles`. Give it a fixed height or max-height. `showScrollToTop` fades in a button once the content is scrolled 200px.",
      },
    },
  },
  argTypes: {
    orientation: {
      control: "inline-radio",
      options: ["vertical", "horizontal", "both"],
    },
    showScrollToTop: { control: "boolean" },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

const RUNS = Array.from({ length: 40 }, (_, index) => index + 1);

function RunList({ detailed = false }: { detailed?: boolean }) {
  return (
    <ul className="flex flex-col">
      {RUNS.map((run) => (
        <li key={run} className="border-b border-zinc-100">
          <a
            href={`#run-${run}`}
            className="block whitespace-nowrap px-4 py-2 hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-400"
          >
            <Text variant="body" as="span">
              Run #{run}
              {detailed
                ? " · Daily digest · completed in 42 seconds · 3 outputs · triggered by schedule"
                : null}
            </Text>
          </a>
        </li>
      ))}
    </ul>
  );
}

export const Vertical: Story = {
  render: function VerticalStory(args) {
    return (
      <ScrollArea
        {...args}
        className="h-72 w-64 rounded-xl border border-zinc-200 bg-white"
      >
        <RunList />
      </ScrollArea>
    );
  },
};

export const Horizontal: Story = {
  args: { orientation: "horizontal" },
  render: function HorizontalStory(args) {
    return (
      <ScrollArea
        {...args}
        className="w-80 rounded-xl border border-zinc-200 bg-white"
      >
        <div className="flex w-max gap-3 p-4">
          {Array.from({ length: 12 }, (_, index) => (
            <a
              key={index}
              href={`#integration-${index + 1}`}
              className="flex h-20 w-28 shrink-0 items-center justify-center rounded-lg bg-zinc-100 hover:bg-zinc-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2"
            >
              <Text variant="small" as="span" tone="secondary">
                Integration {index + 1}
              </Text>
            </a>
          ))}
        </div>
      </ScrollArea>
    );
  },
};

export const Both: Story = {
  args: { orientation: "both" },
  render: function BothStory(args) {
    return (
      <ScrollArea
        {...args}
        className="h-64 w-80 rounded-xl border border-zinc-200 bg-white"
      >
        <RunList detailed />
      </ScrollArea>
    );
  },
};

export const WithScrollToTop: Story = {
  args: { showScrollToTop: true },
  render: function WithScrollToTopStory(args) {
    return (
      <ScrollArea
        {...args}
        className="h-72 w-64 rounded-xl border border-zinc-200 bg-white"
      >
        <RunList />
      </ScrollArea>
    );
  },
};
