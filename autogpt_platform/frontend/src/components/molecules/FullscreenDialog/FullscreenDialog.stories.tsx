import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { fn, userEvent, within } from "storybook/test";
import { FullscreenDialog } from "./FullscreenDialog";

const meta = {
  title: "Molecules/FullscreenDialog",
  component: FullscreenDialog,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "A modal that covers the whole viewport, used for mobile editors. It is always open while mounted: render it conditionally and unmount it from `onClose`, which fires on Escape. `title` is the dialog's accessible name (visually hidden), so `children` supply the visible header and a close control. Focus is trapped inside and returns to the element that opened it.",
      },
      story: { inline: false, iframeHeight: 480 },
    },
  },
  argTypes: {
    title: {
      control: "text",
      description: "Accessible name of the dialog (visually hidden)",
    },
  },
  args: {
    title: "Edit Maria's soul",
    onClose: fn(),
    children: <DialogBody title="Edit Maria's soul" />,
  },
} satisfies Meta<typeof FullscreenDialog>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: renderWithTrigger,
};

export const Open: Story = {
  parameters: { layout: "fullscreen" },
};

export const OpenedFromTrigger: Story = {
  render: renderWithTrigger,
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Edit soul" }),
    );
  },
};

export const LongContent: Story = {
  args: {
    title: "Edit team identities",
    children: <DialogBody title="Edit team identities" sectionCount={12} />,
  },
  parameters: { layout: "fullscreen" },
};

interface DialogBodyProps {
  title: string;
  onClose?: () => void;
  sectionCount?: number;
}

function DialogBody({ title, onClose, sectionCount = 1 }: DialogBodyProps) {
  const sections = Array.from(
    { length: sectionCount },
    (_, index) => index + 1,
  );

  return (
    <>
      <header className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <Text variant="large-medium" as="span">
          {title}
        </Text>
        <Button
          variant="icon"
          size="icon-sm"
          aria-label="Close"
          withTooltip={false}
          onClick={onClose}
        >
          <Icon icon={Cancel01Icon} size={16} />
        </Button>
      </header>
      <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-4">
        {sections.map((section) => (
          <div key={section} className="flex flex-col gap-2">
            <Input
              id={`identity-${section}`}
              label={sectionCount > 1 ? `Identity ${section}` : "Identity"}
              placeholder="A patient research assistant"
              wrapperClassName="mb-0!"
            />
            <Text variant="small" tone="muted">
              Describe who this expert is and how they speak.
            </Text>
          </div>
        ))}
      </div>
      <footer className="flex justify-end gap-2 border-t border-zinc-200 px-4 py-3">
        <Button variant="secondary" size="small" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="primary" size="small" onClick={onClose}>
          Save
        </Button>
      </footer>
    </>
  );
}

function DialogWithTrigger() {
  const [open, setOpen] = useState(false);

  function handleClose() {
    setOpen(false);
  }

  return (
    <>
      <Button variant="primary" size="small" onClick={() => setOpen(true)}>
        Edit soul
      </Button>
      {open ? (
        <FullscreenDialog title="Edit Maria's soul" onClose={handleClose}>
          <DialogBody title="Edit Maria's soul" onClose={handleClose} />
        </FullscreenDialog>
      ) : null}
    </>
  );
}

function renderWithTrigger() {
  return <DialogWithTrigger />;
}
