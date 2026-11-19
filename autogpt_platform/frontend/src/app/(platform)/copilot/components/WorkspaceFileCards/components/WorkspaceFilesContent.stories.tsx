import type { Meta, StoryObj } from "@storybook/nextjs";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { WorkspaceFilesContent } from "./WorkspaceFilesContent";

const FILES: SessionFile[] = [
  {
    messageID: null,
    item: {
      id: "cccccccc-0000-0000-0000-000000000001",
      name: "launch-brief.md",
      path: "/sessions/story/launch-brief.md",
      mime_type: "text/markdown",
      size_bytes: 2048,
      origin: "generated",
      created_at: "2026-10-20T10:00:00Z",
    },
  },
  {
    messageID: null,
    item: {
      id: "cccccccc-0000-0000-0000-000000000002",
      name: "pricing-table.csv",
      path: "/sessions/story/pricing-table.csv",
      mime_type: "text/csv",
      size_bytes: 512,
      origin: "generated",
      created_at: "2026-10-20T10:05:00Z",
    },
  },
  {
    messageID: null,
    item: {
      id: "cccccccc-0000-0000-0000-000000000003",
      name: "competitor-notes.pdf",
      path: "/sessions/story/competitor-notes.pdf",
      mime_type: "application/pdf",
      size_bytes: 184320,
      origin: "uploaded",
      created_at: "2026-10-20T09:40:00Z",
    },
  },
];

const meta: Meta<typeof WorkspaceFilesContent> = {
  title: "Copilot/WorkspaceFilesContent",
  component: WorkspaceFilesContent,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "The files list inside the workspace files card, with skeleton lines while it loads.",
      },
    },
  },
  args: {
    files: [],
    isLoading: false,
    isError: false,
    isZipping: false,
    onOpen: () => undefined,
    onDownload: () => undefined,
    onRequestDelete: () => undefined,
    onDownloadAll: () => undefined,
  },
  decorators: [
    (Story) => (
      <div className="w-[340px] rounded-large border border-border-base bg-surface p-3">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Loading: Story = {
  name: "Loading (skeleton lines)",
  args: { isLoading: true },
};

export const Loaded: Story = {
  args: { files: FILES },
};

export const LoadedDetailedRows: Story = {
  name: "Loaded, detailed rows",
  args: { files: FILES, rowStyle: "detailed", showHeader: false },
};
