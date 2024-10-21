import type { Meta, StoryObj } from "@storybook/react";
import { BreadCrumbs } from "./BreadCrumbs";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/BreadCrumbs",
  component: BreadCrumbs,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    items: { control: "object" },
  },
} satisfies Meta<typeof BreadCrumbs>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "Agents", link: "/agents" },
      { name: "SEO Optimizer", link: "/agents/seo-optimizer" },
    ],
  },
};

export const SingleItem: Story = {
  args: {
    items: [{ name: "Home", link: "/" }],
  },
};

export const LongPath: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "Categories", link: "/categories" },
      { name: "AI Tools", link: "/categories/ai-tools" },
      { name: "Data Analysis", link: "/categories/ai-tools/data-analysis" },
      {
        name: "Data Analyzer",
        link: "/categories/ai-tools/data-analysis/data-analyzer",
      },
    ],
  },
};

export const WithInteraction: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "Agents", link: "/agents" },
      { name: "Task Planner", link: "/agents/task-planner" },
    ],
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const homeLink = canvas.getByText("Home");

    await userEvent.hover(homeLink);
    await userEvent.click(homeLink);
  },
};

export const LongNames: Story = {
  args: {
    items: [
      { name: "Home", link: "/" },
      { name: "AI-Powered Writing Assistants", link: "/ai-writing-assistants" },
      {
        name: "Advanced Grammar and Style Checker",
        link: "/ai-writing-assistants/grammar-style-checker",
      },
    ],
  },
};
