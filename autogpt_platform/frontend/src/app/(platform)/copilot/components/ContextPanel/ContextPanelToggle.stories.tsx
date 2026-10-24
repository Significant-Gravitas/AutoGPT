import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { useCopilotUIStore } from "../../store";
import { ContextPanelToggle } from "./ContextPanelToggle";

const GENERATED_FILES: WorkspaceFileItem[] = [
  {
    id: "aaaaaaaa-0000-0000-0000-000000000001",
    name: "launch-brief.md",
    path: "/sessions/story/launch-brief.md",
    mime_type: "text/markdown",
    size_bytes: 2048,
    origin: "generated",
    created_at: "2026-10-20T10:00:00Z",
  },
  {
    id: "aaaaaaaa-0000-0000-0000-000000000002",
    name: "pricing-table.csv",
    path: "/sessions/story/pricing-table.csv",
    mime_type: "text/csv",
    size_bytes: 512,
    origin: "generated",
    created_at: "2026-10-20T10:05:00Z",
  },
  {
    id: "aaaaaaaa-0000-0000-0000-000000000003",
    name: "summary.html",
    path: "/sessions/story/summary.html",
    mime_type: "text/html",
    size_bytes: 4096,
    origin: "generated",
    created_at: "2026-10-20T10:10:00Z",
  },
];

function filesHandler(files: WorkspaceFileItem[]) {
  return http.get("*/api/workspace/files", () =>
    HttpResponse.json({ files, offset: 0, has_more: false }),
  );
}

interface PanelState {
  isOpen: boolean;
  isComputerOpen: boolean;
}

function setPanel({ isOpen, isComputerOpen }: PanelState) {
  useCopilotUIStore.setState((s) => ({
    artifactPanel: {
      ...s.artifactPanel,
      isOpen,
      activeTab: "files",
      activeArtifact: null,
      history: [],
      lastArtifact: null,
      mode: isComputerOpen ? "computer" : "artifact",
      isComputerOpen,
      computer: null,
    },
  }));
}

const meta: Meta<typeof ContextPanelToggle> = {
  title: "Copilot/ContextPanelToggle",
  component: ContextPanelToggle,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "The chat's top-right controls: the files toggle and the Computer toggle. Both hide below the lg breakpoint, where the real component renders nothing.",
      },
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="flex justify-end bg-surface-muted p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const NoDocuments: Story = {
  name: "Files: 0 documents, both panels closed",
  args: { sessionId: "story-session-empty" },
  decorators: [
    (Story) => {
      setPanel({ isOpen: false, isComputerOpen: false });
      return <Story />;
    },
  ],
  parameters: { msw: { handlers: [filesHandler([])] } },
};

export const ThreeDocuments: Story = {
  name: "Files: 3 documents, both panels closed",
  args: { sessionId: "story-session-three" },
  decorators: [
    (Story) => {
      setPanel({ isOpen: false, isComputerOpen: false });
      return <Story />;
    },
  ],
  parameters: { msw: { handlers: [filesHandler(GENERATED_FILES)] } },
};

export const FilesPanelOpen: Story = {
  name: "Files panel open (3 documents)",
  args: { sessionId: "story-session-files-open" },
  decorators: [
    (Story) => {
      setPanel({ isOpen: true, isComputerOpen: false });
      return <Story />;
    },
  ],
  parameters: { msw: { handlers: [filesHandler(GENERATED_FILES)] } },
};

export const ComputerPanelOpen: Story = {
  name: "Computer panel open",
  args: { sessionId: "story-session-computer-open" },
  decorators: [
    (Story) => {
      setPanel({ isOpen: true, isComputerOpen: true });
      return <Story />;
    },
  ],
  parameters: { msw: { handlers: [filesHandler([])] } },
};
