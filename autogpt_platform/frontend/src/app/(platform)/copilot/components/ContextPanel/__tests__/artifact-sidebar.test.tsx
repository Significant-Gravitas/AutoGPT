import React from "react";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactRef } from "../../../store";
import { useCopilotUIStore } from "../../../store";
import { ArtifactPanel } from "../../ArtifactPanel/ArtifactPanel";
import { ContextPanel } from "../ContextPanel";

const ARTIFACT_ID = "11111111-0000-0000-0000-000000000000";
const ARTIFACT_SOURCE_URL = `/api/proxy/api/workspace/files/${ARTIFACT_ID}/download`;

function makeArtifact(): ArtifactRef {
  return {
    id: ARTIFACT_ID,
    title: "notes.txt",
    mimeType: "text/plain",
    sourceUrl: ARTIFACT_SOURCE_URL,
    origin: "agent",
  };
}

function resetCopilotStore() {
  useCopilotUIStore.setState((s) => ({
    artifactPanel: {
      ...s.artifactPanel,
      isOpen: true,
      width: 600,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      expandedPanel: "context",
    },
  }));
}

// framer-motion's AnimatePresence/motion mount/exit animations make panel
// swaps async and flaky in jsdom. Render them as plain divs so the swap is
// synchronous — same approach as ChatContainer.test.tsx.
vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
  motion: {
    div: React.forwardRef(function MotionDiv(
      props: Record<string, unknown>,
      ref: React.Ref<HTMLDivElement>,
    ) {
      const {
        children,
        initial: _initial,
        animate: _animate,
        exit: _exit,
        transition: _transition,
        ...rest
      } = props as {
        children?: React.ReactNode;
        initial?: unknown;
        animate?: unknown;
        exit?: unknown;
        transition?: unknown;
        [key: string]: unknown;
      };
      return (
        <div ref={ref} {...rest}>
          {children}
        </div>
      );
    }),
  },
}));

function RightRegion({ sessionId }: { sessionId: string }) {
  return (
    <div className="flex">
      <ContextPanel sessionId={sessionId} />
      <ArtifactPanel />
    </div>
  );
}

describe("Context/Artifact panel swap (desktop)", () => {
  beforeEach(() => {
    server.use(
      getListWorkspaceFilesMockHandler200({
        files: [],
        offset: 0,
        has_more: false,
      }),
    );
    // The text artifact fetches its content from the download proxy URL,
    // which isn't an Orval endpoint — stub global fetch for that URL only, and
    // throw on anything else so a stray fetch fails loudly instead of being
    // silently masked.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes(ARTIFACT_SOURCE_URL)) {
          return new Response("hello world", {
            status: 200,
            headers: { "Content-Type": "text/plain" },
          });
        }
        throw new Error(`Unexpected fetch in test: ${url}`);
      }),
    );
    resetCopilotStore();
  });

  afterEach(() => {
    cleanup();
    resetCopilotStore();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("swaps between the artifact and context panels and closes the artifact", async () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact());

    render(<RightRegion sessionId="session-1" />);

    // 1. Artifact opened: artifact header is visible, context shows a rail.
    expect(await screen.findByText("notes.txt")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Expand workspace panel" }),
    ).toBeDefined();

    // 2. Expand the context panel → tabs visible, artifact collapses to a rail.
    fireEvent.click(
      screen.getByRole("button", { name: "Expand workspace panel" }),
    );
    expect(await screen.findByRole("tablist")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Expand artifact panel" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Expand workspace panel" }),
    ).toBeNull();

    // 3. Expand the artifact panel again → header back, context rail returns.
    fireEvent.click(
      screen.getByRole("button", { name: "Expand artifact panel" }),
    );
    expect(await screen.findByText("notes.txt")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Expand workspace panel" }),
    ).toBeDefined();

    // 4. Close the artifact → artifact gone, context panel expanded (no rails).
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(await screen.findByRole("tablist")).toBeDefined();
    expect(screen.queryByText("notes.txt")).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Expand workspace panel" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Expand artifact panel" }),
    ).toBeNull();
  });
});
