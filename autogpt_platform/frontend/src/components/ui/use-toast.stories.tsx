import React from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "./button";
import { useToast } from "./use-toast";
import { Toaster } from "./toaster";

const meta = {
  title: "UI/UseToast",
  component: () => null, // UseToast is a hook, not a component
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <>
        <Story />
        <Toaster />
      </>
    ),
  ],
} satisfies Meta<typeof useToast>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <Button onClick={() => toast({ title: "Default Toast" })}>
        Show Default Toast
      </Button>
    );
  },
};

export const WithDescription: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <Button
        onClick={() =>
          toast({
            title: "Toast with Description",
            description: "This is a more detailed toast message.",
          })
        }
      >
        Show Toast with Description
      </Button>
    );
  },
};

export const Destructive: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <Button
        onClick={() =>
          toast({
            variant: "destructive",
            title: "Destructive Toast",
            description: "This action cannot be undone.",
          })
        }
      >
        Show Destructive Toast
      </Button>
    );
  },
};

export const WithAction: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <Button
        onClick={() =>
          toast({
            title: "Toast with Action",
            description: "Click the action button to do something.",
            action: <Button variant="outline">Action</Button>,
          })
        }
      >
        Show Toast with Action
      </Button>
    );
  },
};

export const CustomDuration: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <Button
        onClick={() =>
          toast({
            title: "Custom Duration Toast",
            description: "This toast will disappear after 5 seconds.",
            duration: 5000,
          })
        }
      >
        Show Custom Duration Toast
      </Button>
    );
  },
};
