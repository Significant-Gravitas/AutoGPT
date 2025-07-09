import { Button } from "@/components/atoms/Button/Button";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { toast } from "sonner";
import { Toast } from "./Toast";

const meta: Meta<typeof Toast> = {
  title: "Molecules/Toast",
  component: Toast,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A Toast component built with Sonner that provides beautiful, customizable notifications. Supports different types (success, error, info, warning, loading), themes, positioning, and rich content. The component handles all toast notifications globally when placed in your app.",
      },
    },
  },
  argTypes: {
    position: {
      control: "select",
      options: ["top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"],
      description: "Position of the toast notifications",
    },
    theme: {
      control: "select",
      options: ["light", "dark", "system"],
      description: "Theme of the toast notifications",
    },
    richColors: {
      control: "boolean",
      description: "Enable rich colors for different toast types",
    },
    expand: {
      control: "boolean",
      description: "Whether toasts should expand on hover",
    },
    duration: {
      control: "number",
      description: "Default duration in milliseconds",
    },
    closeButton: {
      control: "boolean",
      description: "Show close button on toasts",
    },
    visibleToasts: {
      control: "number",
      description: "Maximum number of toasts visible at once",
    },
    className: {
      control: "text",
      description: "CSS class name for styling",
    },
  },
  args: {
    position: "top-right",
    theme: "system",
    richColors: true,
    expand: false,
    duration: 4000,
    closeButton: false,
    visibleToasts: 3,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: renderToastWithButtons,
};

export const Success: Story = {
  render: renderSuccessToast,
};

export const Error: Story = {
  render: renderErrorToast,
};

export const Info: Story = {
  render: renderInfoToast,
};

export const Warning: Story = {
  render: renderWarningToast,
};

export const Loading: Story = {
  render: renderLoadingToast,
};

export const Custom: Story = {
  render: renderCustomToast,
};

export const WithCloseButton: Story = {
  args: {
    closeButton: true,
  },
  render: renderToastWithButtons,
};

export const DifferentPositions: Story = {
  render: renderDifferentPositions,
};

export const DarkTheme: Story = {
  args: {
    theme: "dark",
  },
  render: renderToastWithButtons,
};

export const LightTheme: Story = {
  args: {
    theme: "light",
  },
  render: renderToastWithButtons,
};

export const WithExpand: Story = {
  args: {
    expand: true,
  },
  render: renderToastWithButtons,
};

export const LongDuration: Story = {
  args: {
    duration: 10000,
  },
  render: renderToastWithButtons,
};

function renderToastWithButtons(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" onClick={() => toast.success("Success! Operation completed.")}>
          Success Toast
        </Button>
        <Button variant="secondary" onClick={() => toast.error("Error! Something went wrong.")}>
          Error Toast
        </Button>
        <Button variant="ghost" onClick={() => toast.info("Info: Here's some information.")}>
          Info Toast
        </Button>
        <Button variant="primary" onClick={() => toast.warning("Warning! Please check this.")}>
          Warning Toast
        </Button>
      </div>
    </div>
  );
}

function renderSuccessToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="primary" 
        onClick={() => toast.success("Success! Your changes have been saved.", {
          description: "All data has been successfully synchronized.",
        })}
      >
        Show Success Toast
      </Button>
    </div>
  );
}

function renderErrorToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="secondary" 
        onClick={() => toast.error("Error! Failed to save changes.", {
          description: "Please try again or contact support.",
        })}
      >
        Show Error Toast
      </Button>
    </div>
  );
}

function renderInfoToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="ghost" 
        onClick={() => toast.info("New update available!", {
          description: "Version 2.0.0 is now available for download.",
        })}
      >
        Show Info Toast
      </Button>
    </div>
  );
}

function renderWarningToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="primary" 
        onClick={() => toast.warning("Storage almost full!", {
          description: "You have used 90% of your storage space.",
        })}
      >
        Show Warning Toast
      </Button>
    </div>
  );
}

function renderLoadingToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="primary" 
        onClick={() => {
          const id = toast.loading("Saving changes...", {
            description: "Please wait while we sync your data.",
          });
          
          setTimeout(() => {
            toast.success("Changes saved successfully!", { id });
          }, 3000);
        }}
      >
        Show Loading Toast
      </Button>
    </div>
  );
}

function renderCustomToast(args: any) {
  return (
    <div className="space-y-4">
      <Toast {...args} />
      <Button 
        variant="primary" 
        onClick={() => {
          toast.custom((t) => (
            <div className="flex items-center gap-3 p-4 bg-white border rounded-lg shadow-lg">
              <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold">
                🎉
              </div>
              <div>
                <div className="font-semibold">Custom Toast!</div>
                <div className="text-sm text-gray-600">This is a custom toast with rich content.</div>
              </div>
              <Button size="small" onClick={() => toast.dismiss(t)}>
                Close
              </Button>
            </div>
          ));
        }}
      >
        Show Custom Toast
      </Button>
    </div>
  );
}

function renderDifferentPositions(args: any) {
  const positions = [
    { label: "Top Left", value: "top-left" },
    { label: "Top Right", value: "top-right" },
    { label: "Bottom Left", value: "bottom-left" },
    { label: "Bottom Right", value: "bottom-right" },
    { label: "Top Center", value: "top-center" },
    { label: "Bottom Center", value: "bottom-center" },
  ];

  return (
    <div className="space-y-4">
      <Toast {...args} />
      <div className="grid grid-cols-2 gap-2">
        {positions.map((pos) => (
          <Button
            key={pos.value}
            variant="ghost"
            size="small"
            onClick={() => {
              toast.success(`Toast from ${pos.label}!`, {
                position: pos.value as any,
              });
            }}
          >
            {pos.label}
          </Button>
        ))}
      </div>
    </div>
  );
}