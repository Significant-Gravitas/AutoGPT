import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Alert, AlertTitle, AlertDescription } from "./Alert";

const meta = {
  title: "Molecules/Alert",
  component: Alert,
  parameters: {
    a11y: { test: "error" },
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Alert>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: (
      <>
        <AlertTitle>Heads up!</AlertTitle>
        <AlertDescription>
          You can add components to your app using the cli.
        </AlertDescription>
      </>
    ),
  },
};

export const Warning: Story = {
  args: {
    variant: "warning",
    children: (
      <>
        <AlertTitle>Warning</AlertTitle>
        <AlertDescription>
          This action cannot be undone. Please proceed with caution.
        </AlertDescription>
      </>
    ),
  },
};

export const Error: Story = {
  args: {
    variant: "error",
    children: (
      <>
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          Your session has expired. Please log in again.
        </AlertDescription>
      </>
    ),
  },
};

export const Success: Story = {
  args: {
    variant: "success",
    children: (
      <>
        <AlertTitle>Saved</AlertTitle>
        <AlertDescription>Your agent is published.</AlertDescription>
      </>
    ),
  },
};

export const Info: Story = {
  args: {
    variant: "info",
    children: (
      <>
        <AlertTitle>Scheduled maintenance</AlertTitle>
        <AlertDescription>
          Runs pause for ten minutes on Sunday at 02:00 UTC.
        </AlertDescription>
      </>
    ),
  },
};
