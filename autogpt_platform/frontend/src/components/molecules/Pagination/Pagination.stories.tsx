import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Pagination } from "./Pagination";

const meta: Meta<typeof Pagination> = {
  title: "Molecules/Pagination",
  component: Pagination,
  tags: ["autodocs"],
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
    docs: {
      description: {
        component:
          'Previous and next buttons around page numbers, with an ellipsis for skipped ranges. Controlled through `page`, `pageCount` and `onPageChange`; the current page gets `aria-current="page"`.',
      },
    },
  },
  argTypes: {
    page: { control: { type: "number", min: 1 } },
    pageCount: { control: { type: "number", min: 1 } },
    siblingCount: { control: { type: "number", min: 0 } },
    disabled: { control: "boolean" },
    onPageChange: { action: "pageChange" },
  },
  args: {
    page: 1,
    pageCount: 10,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

function Interactive(props: React.ComponentProps<typeof Pagination>) {
  const [page, setPage] = useState(props.page);
  return (
    <Pagination
      {...props}
      page={page}
      onPageChange={(next) => {
        setPage(next);
        props.onPageChange(next);
      }}
    />
  );
}

export const FirstPage: Story = {
  render: (args) => <Interactive {...args} />,
};

export const MiddlePage: Story = {
  args: { page: 5 },
  render: (args) => <Interactive {...args} />,
};

export const LastPage: Story = {
  args: { page: 10 },
  render: (args) => <Interactive {...args} />,
};

export const FewPages: Story = {
  args: { page: 2, pageCount: 4 },
  render: (args) => <Interactive {...args} />,
};

export const SinglePage: Story = {
  args: { page: 1, pageCount: 1 },
};

export const WideSiblings: Story = {
  args: { page: 20, pageCount: 40, siblingCount: 2 },
  render: (args) => <Interactive {...args} />,
};

export const Disabled: Story = {
  args: { page: 3, disabled: true },
};
