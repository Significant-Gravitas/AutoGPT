import type { Meta, StoryObj } from "@storybook/react";

import {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
} from "./card";

const meta = {
  title: "UI/Card",
  component: Card,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    // Add any specific controls for Card props here if needed
  },
} satisfies Meta<typeof Card>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <CardHeader>
          <CardTitle>Card Title</CardTitle>
          <CardDescription>Card Description</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Card Content</p>
        </CardContent>
        <CardFooter>
          <p>Card Footer</p>
        </CardFooter>
      </>
    ),
  },
};

export const HeaderOnly: Story = {
  args: {
    children: (
      <CardHeader>
        <CardTitle>Header Only Card</CardTitle>
        <CardDescription>This card has only a header.</CardDescription>
      </CardHeader>
    ),
  },
};

export const ContentOnly: Story = {
  args: {
    children: (
      <CardContent>
        <p>This card has only content.</p>
      </CardContent>
    ),
  },
};

export const FooterOnly: Story = {
  args: {
    children: (
      <CardFooter>
        <p>This card has only a footer.</p>
      </CardFooter>
    ),
  },
};

export const CustomContent: Story = {
  args: {
    children: (
      <>
        <CardHeader>
          <CardTitle>Custom Content</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-40 items-center justify-center rounded-md bg-gray-100">
            <span role="img" aria-label="Rocket" style={{ fontSize: "3rem" }}>
              🚀
            </span>
          </div>
        </CardContent>
        <CardFooter className="justify-between">
          <button className="rounded bg-blue-500 px-4 py-2 text-white">
            Action
          </button>
          <p>Footer text</p>
        </CardFooter>
      </>
    ),
  },
};
