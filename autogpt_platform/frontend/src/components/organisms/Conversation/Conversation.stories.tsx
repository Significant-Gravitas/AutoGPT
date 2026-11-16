import { Icon } from "@/components/atoms/Icon/Icon";
import { BubbleChatIcon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { expect, waitFor, within } from "storybook/test";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "./Conversation";
import { Message, MessageContent, MessageResponse } from "../Message/Message";

const TURNS = Array.from({ length: 12 }, (_, index) => ({
  id: `turn-${index + 1}`,
  from: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
  text:
    index % 2 === 0
      ? `Question ${index / 2 + 1}: what should the launch brief say about pricing?`
      : `Answer ${(index + 1) / 2}: keep the starter tier at its current price, call out the annual discount, and list the two open questions so the team can settle them before the launch review.`,
}));

const meta: Meta<typeof Conversation> = {
  title: "AI Elements/Conversation",
  component: Conversation,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Scrolling chat log that sticks to the newest message. The scroll button appears once the reader scrolls away from the bottom.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const ScrolledUp: Story = {
  name: "Scrolled up (scroll button visible)",
  render: () => (
    <div className="flex h-[420px] max-w-2xl flex-col rounded-large border border-border-base">
      <Conversation className="min-h-0 flex-1">
        <ConversationContent>
          {TURNS.map((turn) => (
            <Message from={turn.from} key={turn.id}>
              <MessageContent>
                <MessageResponse>{turn.text}</MessageResponse>
              </MessageContent>
            </Message>
          ))}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const log = await canvas.findByRole("log");
    const scroller = log.firstElementChild as HTMLElement;
    await waitFor(() =>
      expect(scroller.scrollHeight).toBeGreaterThan(scroller.clientHeight),
    );
    // Moving up from the bottom is what releases the stick-to-bottom lock.
    await waitFor(
      () => {
        scroller.scrollTop = 0;
        expect(log.querySelector("button")).not.toBeNull();
      },
      { timeout: 5000 },
    );
  },
};

export const EmptyState: Story = {
  name: "Empty state",
  render: () => (
    <div className="flex h-[420px] max-w-2xl flex-col rounded-large border border-border-base">
      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="h-full">
          <ConversationEmptyState
            title="No messages yet"
            description="Ask Otto anything to start the conversation"
            icon={<Icon icon={BubbleChatIcon} size={32} />}
          />
        </ConversationContent>
      </Conversation>
    </div>
  ),
};
