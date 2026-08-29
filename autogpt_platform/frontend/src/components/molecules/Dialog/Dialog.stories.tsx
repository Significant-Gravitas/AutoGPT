import { Button } from "@/components/atoms/Button/Button";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Dialog } from "./Dialog";

const meta: Meta<typeof Dialog> = {
  title: "Molecules/Dialog",
  component: Dialog,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A responsive dialog component that automatically switches between modal dialog (desktop) and drawer (mobile). Built on top of Radix UI Dialog and Vaul drawer with custom styling. Supports compound components: Dialog.Trigger, Dialog.Content, and Dialog.Footer.",
      },
    },
  },
  argTypes: {
    title: {
      control: "text",
      description: "Dialog title - can be string or React node",
    },
    forceOpen: {
      control: "boolean",
      description: "Force the dialog to stay open (useful for previewing)",
    },
    styling: {
      control: "object",
      description: "Custom CSS styles object",
    },
    onClose: {
      action: "closed",
      description: "Callback fired when dialog closes",
    },
  },
  args: {
    title: "Dialog Title",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: renderBasicDialog,
};

export const WithoutTitle: Story = {
  render: renderDialogWithoutTitle,
};

export const ForceOpen: Story = {
  args: {
    forceOpen: true,
    title: "Preview Dialog",
  },
  render: renderForceOpenDialog,
};

export const WithFooter: Story = {
  render: renderDialogWithFooter,
};

export const Controlled: Story = {
  render: renderControlledDialog,
};

export const CustomStyling: Story = {
  render: renderCustomStyledDialog,
};

export const ModalOverModal: Story = {
  render: renderModalOverModal,
};

function renderBasicDialog() {
  return (
    <Dialog title="Basic Dialog">
      <Dialog.Trigger>
        <Button variant="primary">Open Dialog</Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <p>This is a basic dialog with some content.</p>
        <p>
          It automatically adapts to screen size - modal on desktop, drawer on
          mobile.
        </p>
      </Dialog.Content>
    </Dialog>
  );
}

function renderDialogWithoutTitle() {
  return (
    <Dialog>
      <Dialog.Trigger>
        <Button variant="primary">Open Dialog (no title)</Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <p className="flex min-h-[100px] flex-row items-center justify-center">
          This dialog doesn&apos;t use the title prop, allowing for no header or
          a custom header.
        </p>
      </Dialog.Content>
    </Dialog>
  );
}

function renderForceOpenDialog(args: any) {
  return (
    <Dialog {...args}>
      <Dialog.Content>
        <p>This dialog is forced open for preview purposes.</p>
        <p>
          In real usage, you&apos;d typically use a trigger or controlled state.
        </p>
      </Dialog.Content>
    </Dialog>
  );
}

function renderDialogWithFooter() {
  return (
    <Dialog title="Dialog with Footer">
      <Dialog.Trigger>
        <Button variant="primary">Open Dialog with Footer</Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <p>This dialog includes a footer with action buttons.</p>
        <p>Use the footer for primary and secondary actions.</p>
        <Dialog.Footer>
          <Button variant="ghost" size="small">
            Cancel
          </Button>
          <Button variant="primary" size="small">
            Confirm
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

function renderControlledDialog() {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = async () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="space-y-4">
      <Button variant="primary" onClick={handleToggle}>
        {isOpen ? "Close" : "Open"} Controlled Dialog
      </Button>

      <Dialog
        title="Controlled Dialog"
        controlled={{
          isOpen,
          set: setIsOpen,
        }}
        onClose={() => console.log("Dialog closed")}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <p>This dialog is controlled by external state.</p>
            <p>
              Open state:{" "}
              <span className="font-bold">{isOpen ? "Open" : "Closed"}</span>
            </p>
          </div>
          <Button onClick={handleToggle} className="mt-8" size="small">
            Close this modal
          </Button>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}

function renderCustomStyledDialog() {
  return (
    <Dialog
      title="Custom Styled Dialog"
      styling={{
        maxWidth: "800px",
        backgroundColor: "rgb(248 250 252)",
        border: "2px solid rgb(59 130 246)",
      }}
    >
      <Dialog.Trigger>
        <Button variant="primary">Open Custom Styled Dialog</Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <p>This dialog has custom styling applied.</p>
        <p>You can customize dimensions, colors, and other CSS properties.</p>
      </Dialog.Content>
    </Dialog>
  );
}

function renderModalOverModal() {
  return (
    <Dialog title="Parent Dialog">
      <Dialog.Trigger>
        <Button variant="primary">Open Parent</Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="space-y-4">
          <p>
            This is the parent dialog. You can open another modal on top of it
            using a nested Dialog.
          </p>

          <Dialog title="Child Dialog">
            <Dialog.Trigger>
              <Button size="small">Open Child Modal</Button>
            </Dialog.Trigger>
            <Dialog.Content>
              <p>
                This child dialog is rendered above the parent. Close it first
                to interact with the parent again.
              </p>
            </Dialog.Content>
          </Dialog>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
