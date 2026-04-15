import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { ArtifactPanel } from "./ArtifactPanel";
import { useCopilotUIStore } from "../../store";
import type { ArtifactRef } from "../../store";

const PROXY_BASE = "/api/proxy/api/workspace/files";

function makeArtifact(overrides?: Partial<ArtifactRef>): ArtifactRef {
  return {
    id: "file-001",
    title: "report.html",
    mimeType: "text/html",
    sourceUrl: `${PROXY_BASE}/file-001/download`,
    origin: "agent",
    ...overrides,
  };
}

function openPanelWith(artifact: ArtifactRef) {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: true,
      isMinimized: false,
      isMaximized: false,
      width: 600,
      activeArtifact: artifact,
      history: [],
    },
  });
}

const meta: Meta<typeof ArtifactPanel> = {
  title: "Copilot/ArtifactPanel",
  component: ArtifactPanel,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Side panel for previewing workspace artifacts. Supports resize, minimize, maximize, and navigation history. Bug: panel auto-opens on chat switch instead of staying collapsed.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="flex h-[600px] w-full">
        <div className="flex-1 bg-zinc-50 p-8">
          <p className="text-sm text-zinc-500">Chat area</p>
        </div>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const OpenWithTextArtifact: Story = {
  name: "Open — Text File",
  decorators: [
    (Story) => {
      openPanelWith(
        makeArtifact({ title: "notes.txt", mimeType: "text/plain" }),
      );
      return <Story />;
    },
  ],
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/file-001/download`, () => {
          return HttpResponse.text(
            "These are some notes from the agent execution.\n\nKey findings:\n1. Performance improved by 23%\n2. Memory usage reduced\n3. Error rate dropped to 0.1%",
          );
        }),
      ],
    },
  },
};

export const OpenWithHTMLArtifact: Story = {
  name: "Open — HTML",
  decorators: [
    (Story) => {
      openPanelWith(
        makeArtifact({
          id: "html-panel",
          title: "dashboard.html",
          mimeType: "text/html",
          sourceUrl: `${PROXY_BASE}/html-panel/download`,
        }),
      );
      return <Story />;
    },
  ],
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/html-panel/download`, () => {
          return HttpResponse.text(
            `<!DOCTYPE html><html><body class="p-8 font-sans"><h1 class="text-2xl font-bold text-indigo-600">Dashboard</h1><p class="mt-2 text-gray-600">HTML artifact in the panel.</p></body></html>`,
          );
        }),
      ],
    },
  },
};

export const OpenWithImageArtifact: Story = {
  name: "Open — Image (Bug: No Loading State)",
  decorators: [
    (Story) => {
      openPanelWith(
        makeArtifact({
          id: "img-panel",
          title: "chart.png",
          mimeType: "image/png",
          sourceUrl: `${PROXY_BASE}/img-panel/download`,
        }),
      );
      return <Story />;
    },
  ],
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/img-panel/download`, () => {
          return HttpResponse.text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="300"><rect width="500" height="300" fill="#dbeafe"/><text x="250" y="150" text-anchor="middle" fill="#1e40af" font-size="20">Image Preview (no skeleton)</text></svg>',
            { headers: { "Content-Type": "image/svg+xml" } },
          );
        }),
      ],
    },
    docs: {
      description: {
        story:
          "**BUG:** Image artifacts render with a bare `<img>` tag — no loading skeleton or error handling. Compare with text/HTML artifacts which show a proper skeleton while loading.",
      },
    },
  },
};

export const MinimizedStrip: Story = {
  name: "Minimized",
  decorators: [
    (Story) => {
      useCopilotUIStore.setState({
        artifactPanel: {
          isOpen: true,
          isMinimized: true,
          isMaximized: false,
          width: 600,
          activeArtifact: makeArtifact(),
          history: [],
        },
      });
      return <Story />;
    },
  ],
};

export const ErrorState: Story = {
  name: "Error — Failed to Load (Stale Artifact)",
  decorators: [
    (Story) => {
      openPanelWith(
        makeArtifact({
          id: "stale-panel",
          title: "old-report.html",
          mimeType: "text/html",
          sourceUrl: `${PROXY_BASE}/stale-panel/download`,
        }),
      );
      return <Story />;
    },
  ],
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/stale-panel/download`, () => {
          return new HttpResponse(null, { status: 404 });
        }),
      ],
    },
    docs: {
      description: {
        story:
          "Shows what users see when opening a previously generated artifact that no longer exists on the backend (404). The 'Try again' button retries the fetch.",
      },
    },
  },
};

export const Closed: Story = {
  name: "Closed (Default State)",
  decorators: [
    (Story) => {
      useCopilotUIStore.setState({
        artifactPanel: {
          isOpen: false,
          isMinimized: false,
          isMaximized: false,
          width: 600,
          activeArtifact: null,
          history: [],
        },
      });
      return <Story />;
    },
  ],
  parameters: {
    docs: {
      description: {
        story:
          "The default state — panel is closed. It should only open when a user clicks on an artifact card in the chat.",
      },
    },
  },
};
