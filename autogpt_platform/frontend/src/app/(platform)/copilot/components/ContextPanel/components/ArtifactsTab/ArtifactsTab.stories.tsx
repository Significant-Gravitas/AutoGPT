import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { ArtifactsTab } from "./ArtifactsTab";

const SESSION_FILES: WorkspaceFileItem[] = [
  {
    id: "bbbbbbbb-0000-0000-0000-000000000001",
    name: "launch-brief.md",
    path: "/sessions/story/launch-brief.md",
    mime_type: "text/markdown",
    size_bytes: 2048,
    origin: "generated",
    created_at: "2026-10-20T10:00:00Z",
  },
  {
    id: "bbbbbbbb-0000-0000-0000-000000000002",
    name: "pricing-table.csv",
    path: "/sessions/story/pricing-table.csv",
    mime_type: "text/csv",
    size_bytes: 512,
    origin: "generated",
    created_at: "2026-10-20T10:05:00Z",
  },
  {
    id: "bbbbbbbb-0000-0000-0000-000000000003",
    name: "competitor-notes.pdf",
    path: "/sessions/story/competitor-notes.pdf",
    mime_type: "application/pdf",
    size_bytes: 184320,
    origin: "uploaded",
    created_at: "2026-10-20T09:40:00Z",
  },
];

const meta: Meta<typeof ArtifactsTab> = {
  title: "Copilot/ArtifactsTab",
  component: ArtifactsTab,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "The side panel's artifacts face: what this chat produced, with skeleton rows while the list loads.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="flex h-[480px] w-[360px] flex-col rounded-large border border-border-base bg-surface">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Loading: Story = {
  name: "Loading (skeleton rows)",
  args: { sessionId: "story-artifacts-loading" },
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/workspace/files", async () => {
          await delay("infinite");
          return HttpResponse.json({ files: [] });
        }),
      ],
    },
  },
};

export const Loaded: Story = {
  args: { sessionId: "story-artifacts-loaded" },
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/workspace/files", () =>
          HttpResponse.json({
            files: SESSION_FILES,
            offset: 0,
            has_more: false,
          }),
        ),
      ],
    },
  },
};

export const Empty: Story = {
  name: "Empty (nothing to preview)",
  args: { sessionId: "story-artifacts-empty" },
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/workspace/files", () =>
          HttpResponse.json({ files: [], offset: 0, has_more: false }),
        ),
      ],
    },
  },
};
