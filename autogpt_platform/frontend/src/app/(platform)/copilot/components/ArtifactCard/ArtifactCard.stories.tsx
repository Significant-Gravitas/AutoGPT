import type { Meta, StoryObj } from "@storybook/nextjs";
import { ArtifactCard } from "./ArtifactCard";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";

function makeArtifact(overrides?: Partial<ArtifactRef>): ArtifactRef {
  return {
    id: "file-001",
    title: "report.html",
    mimeType: "text/html",
    sourceUrl: "/api/proxy/api/workspace/files/file-001/download",
    origin: "agent",
    ...overrides,
  };
}

const meta: Meta<typeof ArtifactCard> = {
  title: "Copilot/ArtifactCard",
  component: ArtifactCard,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Inline artifact card rendered in chat messages. Openable artifacts show a caret and open the ArtifactPanel on click. Download-only artifacts trigger a file download.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="w-96">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const OpenableHTML: Story = {
  name: "Openable (HTML)",
  args: {
    artifact: makeArtifact({
      title: "dashboard.html",
      mimeType: "text/html",
    }),
  },
};

export const OpenableImage: Story = {
  name: "Openable (Image)",
  args: {
    artifact: makeArtifact({
      id: "img-card",
      title: "chart.png",
      mimeType: "image/png",
    }),
  },
};

export const OpenableCode: Story = {
  name: "Openable (Code)",
  args: {
    artifact: makeArtifact({
      title: "script.py",
      mimeType: "text/x-python",
    }),
  },
};

export const DownloadOnly: Story = {
  name: "Download Only (ZIP)",
  args: {
    artifact: makeArtifact({
      title: "archive.zip",
      mimeType: "application/zip",
      sizeBytes: 2_500_000,
    }),
  },
};

export const PreviewableVideo: Story = {
  name: "Previewable (Video)",
  args: {
    artifact: makeArtifact({
      title: "demo.mp4",
      mimeType: "video/mp4",
      sizeBytes: 15_000_000,
    }),
  },
  parameters: {
    docs: {
      description: {
        story:
          "Videos with supported formats (MP4, WebM, M4V) are previewable inline in the artifact panel.",
      },
    },
  },
};

export const WithSize: Story = {
  name: "With File Size",
  args: {
    artifact: makeArtifact({
      title: "data.csv",
      mimeType: "text/csv",
      sizeBytes: 524_288,
    }),
  },
};

export const UserUpload: Story = {
  name: "User Upload Origin",
  args: {
    artifact: makeArtifact({
      title: "requirements.txt",
      mimeType: "text/plain",
      origin: "user-upload",
    }),
  },
};

export const ActiveState: Story = {
  name: "Active (Panel Open)",
  args: {
    artifact: makeArtifact({ id: "active-card" }),
  },
  decorators: [
    (Story) => {
      useCopilotUIStore.setState({
        artifactPanel: {
          isOpen: true,
          isMinimized: false,
          isMaximized: false,
          width: 600,
          activeArtifact: makeArtifact({ id: "active-card" }),
          history: [],
        },
      });
      return <Story />;
    },
  ],
};
