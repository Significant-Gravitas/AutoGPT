import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import {
  InputGroup,
  InputGroupAddon,
} from "@/components/molecules/InputGroup/InputGroup";
import { cn } from "@/lib/utils";
import {
  ArrowUp02Icon,
  GlobalSearchIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ChatStatus } from "ai";
import { useState } from "react";

import {
  PromptInputButton,
  PromptInputSubmit,
  PromptInputTextarea,
} from "./PromptInput";

// The classes ChatInput gives its send and Queue buttons on the default
// (single pill row) composer.
const SEND_BUTTON_CLASS = "max-md:size-10";
const QUEUE_BUTTON_CLASS =
  "size-[2.625rem] rounded-full border-action bg-action text-action-foreground hover:border-action-hover hover:bg-action-hover disabled:border-border-base disabled:bg-surface-active disabled:text-action-foreground disabled:opacity-100";

interface ComposerProps {
  initialValue?: string;
  textareaDisabled?: boolean;
  status?: ChatStatus;
  canStop?: boolean;
  showQueue?: boolean;
}

function Composer({
  initialValue = "",
  textareaDisabled = false,
  status,
  canStop = false,
  showQueue = false,
}: ComposerProps) {
  const [value, setValue] = useState(initialValue);
  const canSend = !textareaDisabled && value.trim().length > 0;

  return (
    <form
      className="relative w-full max-w-3xl pt-16"
      onSubmit={(event) => event.preventDefault()}
    >
      <InputGroup className="relative z-10 flex-col overflow-hidden !rounded-[2rem] border-border-base shadow-sm has-[[data-slot=input-group-control]:focus-visible]:border-border-strong has-[[data-slot=input-group-control]:focus-visible]:ring-0 max-md:!rounded-3xl">
        <div className="flex w-full flex-wrap items-end">
          <InputGroupAddon
            align="inline-start"
            className="relative order-none gap-1 py-1 pl-1.5"
          >
            <PromptInputButton tooltip="Add files" aria-label="Add files">
              <Icon icon={PlusSignIcon} className="size-4" />
            </PromptInputButton>
          </InputGroupAddon>
          <div className="relative min-w-0 flex-1 max-md:order-first max-md:w-full max-md:basis-full">
            <PromptInputTextarea
              aria-label="Message"
              placeholder="Ask Otto anything"
              value={value}
              disabled={textareaDisabled}
              onChange={(event) => setValue(event.target.value)}
            />
          </div>
          <InputGroupAddon
            align="inline-end"
            className="order-none ml-auto gap-1 py-1 pr-1.5"
          >
            <PromptInputButton
              tooltip={{ content: "Search the web", shortcut: "⌘K" }}
            >
              <Icon icon={GlobalSearchIcon} className="size-4" />
              <Text variant="body-medium" as="span" tone="body">
                Search
              </Text>
            </PromptInputButton>
            {showQueue && (
              <PromptInputButton
                aria-label="Queue message"
                tooltip="Queue message"
                variant="default"
                type="button"
                className={cn(QUEUE_BUTTON_CLASS, SEND_BUTTON_CLASS)}
              >
                <Icon icon={ArrowUp02Icon} className="size-4" />
              </PromptInputButton>
            )}
            {status === "submitted" ? (
              <PromptInputSubmit
                status="submitted"
                aria-label="Sending"
                className={SEND_BUTTON_CLASS}
              />
            ) : (
              <PromptInputSubmit
                status={status}
                onStop={canStop ? () => undefined : undefined}
                disabled={status === "streaming" ? false : !canSend}
                className={SEND_BUTTON_CLASS}
              />
            )}
          </InputGroupAddon>
        </div>
      </InputGroup>
    </form>
  );
}

const meta: Meta<typeof Composer> = {
  title: "AI Elements/PromptInput",
  component: Composer,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "PromptInputTextarea, PromptInputButton and PromptInputSubmit composed the way the copilot ChatInput lays them out.",
      },
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  name: "Empty (submit disabled)",
};

export const WithText: Story = {
  name: "With text",
  args: { initialValue: "Draft a one-page launch brief for the new tier" },
};

export const TextareaDisabled: Story = {
  name: "Textarea disabled",
  args: { textareaDisabled: true },
};

export const Submitted: Story = {
  name: "Submitted (spinner)",
  args: {
    initialValue: "Draft a one-page launch brief for the new tier",
    status: "submitted",
  },
};

export const Streaming: Story = {
  name: "Streaming with stop",
  args: { status: "streaming", canStop: true },
};

export const QueueMessage: Story = {
  name: "Streaming with Queue message",
  args: {
    initialValue: "Also add a pricing table",
    status: "streaming",
    canStop: true,
    showQueue: true,
  },
};
