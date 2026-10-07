import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { fn } from "storybook/test";
import { ErrorBoundary } from "./ErrorBoundary";

const meta = {
  title: "Molecules/ErrorBoundary",
  component: ErrorBoundary,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    docs: {
      description: {
        component:
          "Catches render errors in its subtree, reports them to Sentry tagged with `context`, and shows a fallback. Without a `fallback` it renders a full-height `ErrorCard` with the error message and a Try Again button that clears the error and calls `onReset`.",
      },
    },
  },
  argTypes: {
    context: {
      control: "text",
      description: 'Where the error happened. Defaults to "application"',
    },
    fallback: {
      control: false,
      description: "Custom node rendered instead of the default ErrorCard",
    },
  },
  args: {
    context: "agent runs",
    onReset: fn(),
    children: <HealthyWidget />,
  },
} satisfies Meta<typeof ErrorBoundary>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithError: Story = {
  args: { children: <BrokenWidget /> },
  parameters: { layout: "fullscreen" },
};

export const WithoutMessage: Story = {
  args: { children: <BrokenWidgetWithoutMessage /> },
  parameters: { layout: "fullscreen" },
};

export const CustomFallback: Story = {
  args: {
    children: <BrokenWidget />,
    fallback: (
      <Card className="w-80">
        <Text variant="body-medium">Run history is unavailable</Text>
        <Text variant="small" tone="secondary">
          Refresh the page to try again.
        </Text>
      </Card>
    ),
  },
};

export const RecoversOnRetry: Story = {
  render: renderRecoversOnRetry,
  parameters: { layout: "fullscreen" },
};

function HealthyWidget() {
  return (
    <Card className="w-80">
      <Text variant="body-medium">Recent runs</Text>
      <Text variant="small" tone="secondary">
        12 runs completed today.
      </Text>
    </Card>
  );
}

function BrokenWidget(): never {
  throw new Error("Failed to load the agent run history.");
}

function BrokenWidgetWithoutMessage(): never {
  throw new Error("");
}

interface FlakyWidgetProps {
  shouldThrow: boolean;
}

function FlakyWidget({ shouldThrow }: FlakyWidgetProps) {
  if (shouldThrow) {
    throw new Error("The run history request timed out.");
  }
  return <HealthyWidget />;
}

function RecoverableDemo() {
  const [shouldThrow, setShouldThrow] = useState(true);

  return (
    <div className="flex flex-col items-center gap-4 p-8">
      <Button
        variant="secondary"
        size="md"
        onClick={() => setShouldThrow(true)}
      >
        Break the widget
      </Button>
      <ErrorBoundary context="agent runs" onReset={() => setShouldThrow(false)}>
        <FlakyWidget shouldThrow={shouldThrow} />
      </ErrorBoundary>
    </div>
  );
}

function renderRecoversOnRetry() {
  return <RecoverableDemo />;
}
