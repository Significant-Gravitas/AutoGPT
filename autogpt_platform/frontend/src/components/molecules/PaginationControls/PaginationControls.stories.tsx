import type { Meta, StoryObj } from "@storybook/nextjs";
import { PaginationControls } from "./PaginationControls";

const meta: Meta<typeof PaginationControls> = {
  title: "Molecules/PaginationControls",
  component: PaginationControls,
  parameters: {
    layout: "centered",
    nextjs: {
      appDirectory: true,
      navigation: { pathname: "/admin/marketplace" },
    },
  },
};

export default meta;
type Story = StoryObj<typeof PaginationControls>;

export const FirstPage: Story = { args: { currentPage: 1, totalPages: 33 } };
export const MiddlePage: Story = { args: { currentPage: 2, totalPages: 33 } };
export const LastPage: Story = { args: { currentPage: 33, totalPages: 33 } };
export const SinglePage: Story = { args: { currentPage: 1, totalPages: 1 } };
