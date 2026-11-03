import type { Meta, StoryObj } from "@storybook/nextjs";

import {
  Message,
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
  MessageContent,
  MessageResponse,
} from "./Message";

const meta: Meta<typeof MessageBranch> = {
  title: "AI Elements/MessageBranch",
  component: MessageBranch,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Alternate versions of one reply, with previous/next buttons and a page counter.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const TwoBranches: Story = {
  name: "Two branches (buttons enabled)",
  render: () => (
    <div className="max-w-2xl">
      <MessageBranch>
        <MessageBranchContent>
          <Message from="assistant" key="branch-1">
            <MessageContent>
              <MessageResponse>
                {
                  "First version: the launch brief covers goals, audience and the timeline."
                }
              </MessageResponse>
            </MessageContent>
          </Message>
          <Message from="assistant" key="branch-2">
            <MessageContent>
              <MessageResponse>
                {
                  "Second version: a shorter brief that leads with the open pricing questions."
                }
              </MessageResponse>
            </MessageContent>
          </Message>
        </MessageBranchContent>
        <MessageBranchSelector from="assistant">
          <MessageBranchPrevious />
          <MessageBranchPage />
          <MessageBranchNext />
        </MessageBranchSelector>
      </MessageBranch>
    </div>
  ),
};

// The selector hides itself with a single branch, so the buttons render
// directly inside the branch here to show their disabled state.
export const SingleBranch: Story = {
  name: "Single branch (buttons disabled)",
  render: () => (
    <div className="max-w-2xl">
      <MessageBranch>
        <MessageBranchContent>
          <Message from="assistant" key="only-branch">
            <MessageContent>
              <MessageResponse>
                {"The only version of this reply."}
              </MessageResponse>
            </MessageContent>
          </Message>
        </MessageBranchContent>
        <div className="flex items-center gap-1">
          <MessageBranchPrevious />
          <MessageBranchPage />
          <MessageBranchNext />
        </div>
      </MessageBranch>
    </div>
  ),
};
