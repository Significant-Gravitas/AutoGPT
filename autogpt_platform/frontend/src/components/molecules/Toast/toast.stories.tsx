import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { Toaster } from "./toaster";

const meta = {
  title: "Molecules/Toast",
  component: Toaster,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof Toaster>;

export default meta;
type Story = StoryObj<typeof meta>;

// Helper component to demonstrate toast functionality
function ToastDemo() {
  function handleDefaultToast() {
    toast({
      title: "Default Toast",
      description: "This is a default toast message.",
    });
  }

  function handleSuccessToast() {
    toast({
      title: "Success!",
      description: "Your operation was completed successfully.",
      variant: "success",
    });
  }

  function handleDestructiveToast() {
    toast({
      title: "Error!",
      description: "Something went wrong. Please try again.",
      variant: "destructive",
    });
  }

  function handleInfoToast() {
    toast({
      title: "Information",
      description: "Here's some helpful information for you.",
      variant: "info",
    });
  }

  function handleToastWithAction() {
    toast({
      title: "Toast with Action",
      description: "This toast has a custom action button.",
      action: (
        <Button variant="secondary" size="small" className="ml-6">
          Action
        </Button>
      ),
    });
  }

  function handlePersistentToast() {
    toast({
      title: "Persistent Toast",
      description: "This toast won't auto-dismiss.",
      dismissable: false,
    });
  }

  function handleLongDurationToast() {
    toast({
      title: "Long Duration Toast",
      description: "This toast stays visible for 10 seconds.",
      duration: 10000,
    });
  }

  return (
    <div className="flex flex-col gap-4 p-8">
      <h2 className="text-2xl font-bold">Toast Examples</h2>
      <div className="flex flex-wrap gap-4">
        <Button onClick={handleDefaultToast}>Default Toast</Button>
        <Button onClick={handleSuccessToast}>Success Toast</Button>
        <Button onClick={handleDestructiveToast}>Error Toast</Button>
        <Button onClick={handleInfoToast}>Info Toast</Button>
        <Button onClick={handleToastWithAction}>Toast with Action</Button>
        <Button onClick={handlePersistentToast}>Persistent Toast</Button>
        <Button onClick={handleLongDurationToast}>Long Duration Toast</Button>
      </div>
      <Toaster />
    </div>
  );
}

export const Default: Story = {
  render: () => <ToastDemo />,
};

export const SuccessToast: Story = {
  render: () => (
    <div className="p-8">
      <Button
        onClick={() =>
          toast({
            title: "Success!",
            description: "Your operation was completed successfully.",
            variant: "success",
          })
        }
      >
        Show Success Toast
      </Button>
      <Toaster />
    </div>
  ),
};

export const ErrorToast: Story = {
  render: () => (
    <div className="p-8">
      <Button
        onClick={() =>
          toast({
            title: "Error!",
            description: "Something went wrong. Please try again.",
            variant: "destructive",
          })
        }
      >
        Show Error Toast
      </Button>
      <Toaster />
    </div>
  ),
};

export const InfoToast: Story = {
  render: () => (
    <div className="p-8">
      <Button
        onClick={() =>
          toast({
            title: "Information",
            description: "Here's some helpful information for you.",
            variant: "info",
          })
        }
      >
        Show Info Toast
      </Button>
      <Toaster />
    </div>
  ),
};

export const ToastWithAction: Story = {
  render: () => (
    <div className="p-8">
      <Button
        onClick={() =>
          toast({
            title: "Toast with Action",
            description: "This toast has a custom action button.",
            action: (
              <Button variant="secondary" size="small">
                Action
              </Button>
            ),
          })
        }
      >
        Show Toast with Action
      </Button>
      <Toaster />
    </div>
  ),
};

export const PersistentToast: Story = {
  render: () => (
    <div className="p-8">
      <Button
        onClick={() =>
          toast({
            title: "Persistent Toast",
            description: "This toast won't auto-dismiss. Click the X to close.",
            dismissable: false,
          })
        }
      >
        Show Persistent Toast
      </Button>
      <Toaster />
    </div>
  ),
};
