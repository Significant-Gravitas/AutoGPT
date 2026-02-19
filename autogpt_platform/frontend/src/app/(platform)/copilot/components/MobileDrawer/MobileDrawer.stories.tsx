import type { Meta, StoryObj } from "@storybook/nextjs";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { MobileDrawer } from "./MobileDrawer";

const mockSessions: SessionSummaryResponse[] = [
  {
    id: "session-1",
    title: "Help me build a weather agent",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: "session-2",
    title: "Debug my email block",
    created_at: new Date(Date.now() - 86400000).toISOString(),
    updated_at: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: "session-3",
    title: "Search for image generation blocks",
    created_at: new Date(Date.now() - 86400000 * 3).toISOString(),
    updated_at: new Date(Date.now() - 86400000 * 3).toISOString(),
  },
];

const noop = () => {};

const meta: Meta<typeof MobileDrawer> = {
  title: "CoPilot/Chat/MobileDrawer",
  component: MobileDrawer,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Slide-out drawer for mobile showing the list of chat sessions.",
      },
    },
  },
  args: {
    isOpen: true,
    currentSessionId: null,
    isLoading: false,
    sessions: [],
    onSelectSession: noop,
    onNewChat: noop,
    onClose: noop,
    onOpenChange: noop,
  },
};
export default meta;
type Story = StoryObj<typeof MobileDrawer>;

export const Empty: Story = {};

export const WithSessions: Story = {
  args: {
    sessions: mockSessions,
    currentSessionId: "session-1",
  },
};

export const Loading: Story = {
  args: { isLoading: true },
};

export const WithCurrentSession: Story = {
  args: {
    sessions: mockSessions,
    currentSessionId: "session-2",
  },
};
