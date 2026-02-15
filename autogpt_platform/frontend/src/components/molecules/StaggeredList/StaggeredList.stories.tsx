import type { Meta, StoryObj } from "@storybook/nextjs";
import { StaggeredList } from "./StaggeredList";

const meta: Meta<typeof StaggeredList> = {
  title: "Molecules/StaggeredList",
  component: StaggeredList,
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
type Story = StoryObj<typeof StaggeredList>;

const DemoCard = ({ title, index }: { title: string; index: number }) => (
  <div className="rounded-xl bg-neutral-100 p-4 dark:bg-neutral-800">
    <h3 className="mb-1 font-semibold text-neutral-900 dark:text-neutral-100">
      {title}
    </h3>
    <p className="text-sm text-neutral-600 dark:text-neutral-400">
      Card #{index + 1} with staggered animation
    </p>
  </div>
);

const items = ["First Item", "Second Item", "Third Item", "Fourth Item"];

export const Default: Story = {
  args: {
    direction: "up",
    className: "space-y-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const FadeDown: Story = {
  args: {
    direction: "down",
    className: "space-y-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const FadeLeft: Story = {
  args: {
    direction: "left",
    className: "flex gap-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const FadeRight: Story = {
  args: {
    direction: "right",
    className: "flex gap-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const FastStagger: Story = {
  args: {
    direction: "up",
    staggerDelay: 0.05,
    className: "space-y-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const SlowStagger: Story = {
  args: {
    direction: "up",
    staggerDelay: 0.3,
    className: "space-y-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const WithInitialDelay: Story = {
  args: {
    direction: "up",
    initialDelay: 0.5,
    className: "space-y-4",
    children: items.map((item, i) => (
      <DemoCard key={i} title={item} index={i} />
    )),
  },
};

export const GridLayout: Story = {
  args: {
    direction: "up",
    staggerDelay: 0.08,
    className: "grid grid-cols-2 gap-4 md:grid-cols-4",
    children: [
      ...items,
      "Fifth Item",
      "Sixth Item",
      "Seventh Item",
      "Eighth Item",
    ].map((item, i) => <DemoCard key={i} title={item} index={i} />),
  },
};

export const AgentCardsExample: Story = {
  render: () => {
    const agents = [
      { name: "SEO Optimizer", runs: 1234 },
      { name: "Content Writer", runs: 987 },
      { name: "Data Analyzer", runs: 756 },
      { name: "Code Reviewer", runs: 543 },
    ];

    return (
      <StaggeredList
        direction="up"
        staggerDelay={0.1}
        className="grid grid-cols-2 gap-6 md:grid-cols-4"
      >
        {agents.map((agent, i) => (
          <div
            key={i}
            className="rounded-2xl bg-white p-4 shadow-md dark:bg-neutral-900"
          >
            <div className="mb-3 aspect-video rounded-xl bg-gradient-to-br from-violet-500 to-blue-500" />
            <h3 className="mb-1 font-semibold text-neutral-900 dark:text-neutral-100">
              {agent.name}
            </h3>
            <p className="text-sm text-neutral-500">{agent.runs} runs</p>
          </div>
        ))}
      </StaggeredList>
    );
  },
};

export const CreatorCardsExample: Story = {
  render: () => {
    const creators = [
      { name: "Alice", agents: 12 },
      { name: "Bob", agents: 8 },
      { name: "Charlie", agents: 15 },
      { name: "Diana", agents: 6 },
    ];

    const colors = [
      "bg-violet-100 dark:bg-violet-900/30",
      "bg-blue-100 dark:bg-blue-900/30",
      "bg-green-100 dark:bg-green-900/30",
      "bg-orange-100 dark:bg-orange-900/30",
    ];

    return (
      <StaggeredList
        direction="up"
        staggerDelay={0.12}
        className="grid grid-cols-2 gap-6 md:grid-cols-4"
      >
        {creators.map((creator, i) => (
          <div
            key={i}
            className={`rounded-2xl p-5 ${colors[i % colors.length]}`}
          >
            <div className="mb-3 h-12 w-12 rounded-full bg-neutral-300 dark:bg-neutral-700" />
            <h3 className="mb-1 font-semibold text-neutral-900 dark:text-neutral-100">
              {creator.name}
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              {creator.agents} agents
            </p>
          </div>
        ))}
      </StaggeredList>
    );
  },
};
