import type { Meta, StoryObj } from "@storybook/react";

import { Toaster } from "./toaster";
import { useToast } from "./use-toast";
import { Button } from "./button";

const meta = {
  title: "UI/Toaster",
  component: Toaster,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Toaster>;

export default meta;
type Story = StoryObj<typeof meta>;

const ToasterDemo = () => {
  const { toast } = useToast();

  return (
    <div>
      <Button
        onClick={() =>
          toast({
            title: "Toast Title",
            description: "This is a toast description",
          })
        }
      >
        Show Toast
      </Button>
      <Toaster />
    </div>
  );
};

export const Default: Story = {
  render: () => <ToasterDemo />,
};

export const WithTitle: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <div>
        <Button
          onClick={() =>
            toast({
              title: "Toast with Title",
            })
          }
        >
          Show Toast with Title
        </Button>
        <Toaster />
      </div>
    );
  },
};

export const WithDescription: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <div>
        <Button
          onClick={() =>
            toast({
              description: "This is a toast with only a description",
            })
          }
        >
          Show Toast with Description
        </Button>
        <Toaster />
      </div>
    );
  },
};

export const WithAction: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <div>
        <Button
          onClick={() =>
            toast({
              title: "Toast with Action",
              description: "This toast has an action button.",
              action: <Button variant="outline">Action</Button>,
            })
          }
        >
          Show Toast with Action
        </Button>
        <Toaster />
      </div>
    );
  },
};

export const Destructive: Story = {
  render: () => {
    const { toast } = useToast();
    return (
      <div>
        <Button
          onClick={() =>
            toast({
              variant: "destructive",
              title: "Destructive Toast",
              description: "This is a destructive toast message.",
            })
          }
        >
          Show Destructive Toast
        </Button>
        <Toaster />
      </div>
    );
  },
};
