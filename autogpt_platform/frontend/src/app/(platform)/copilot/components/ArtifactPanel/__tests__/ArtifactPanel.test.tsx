import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactRef } from "../../../store";
import {
  DEFAULT_ARTIFACT_PANEL_WIDTH,
  PANEL_RESERVED_WIDTH,
  useCopilotUIStore,
} from "../../../store";
import { ArtifactPanel } from "../ArtifactPanel";

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

class ResizeObserverMock {
  static instances: ResizeObserverMock[] = [];
  callback: ResizeObserverCallback;
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    ResizeObserverMock.instances.push(this);
  }
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
  trigger() {
    this.callback([], this as unknown as ResizeObserver);
  }
}

function getPanel(): HTMLElement {
  const panel = document.querySelector("[data-artifact-panel]");
  if (!(panel instanceof HTMLElement)) throw new Error("panel not rendered");
  return panel;
}

/** The panel tweens its width open, so the inline style passes through
 *  intermediate values before settling on the clamped result. */
async function expectPanelWidth(px: number) {
  await waitFor(() => expect(getPanel().style.width).toBe(`${px}px`));
}

describe("ArtifactPanel (desktop) width clamping", () => {
  let offsetWidthSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    ResizeObserverMock.instances = [];
    vi.stubGlobal("ResizeObserver", ResizeObserverMock);
    // The text artifact fetches its content from the download proxy URL,
    // which isn't an Orval endpoint — stub global fetch for that URL only.
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
    useCopilotUIStore.setState((s) => ({
      artifactPanelWidth: DEFAULT_ARTIFACT_PANEL_WIDTH,
      artifactPanel: {
        ...s.artifactPanel,
        isOpen: true,
        activeArtifact: null,
        history: [],
        activeTab: "files",
      },
    }));
    useCopilotUIStore.getState().openArtifact(makeArtifact());
  });

  afterEach(() => {
    cleanup();
    offsetWidthSpy?.mockRestore();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("shrinks below the stored width when the row leaves less space", async () => {
    offsetWidthSpy = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(1000);

    render(<ArtifactPanel />);

    expect(await screen.findByText("notes.txt")).toBeDefined();
    await expectPanelWidth(1000 - PANEL_RESERVED_WIDTH);
  });

  it("keeps the stored width when the row leaves enough space", async () => {
    offsetWidthSpy = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(2000);

    render(<ArtifactPanel />);

    expect(await screen.findByText("notes.txt")).toBeDefined();
    await expectPanelWidth(DEFAULT_ARTIFACT_PANEL_WIDTH);
  });

  it("re-clamps when the row is resized", async () => {
    offsetWidthSpy = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(2000);

    render(<ArtifactPanel />);
    expect(await screen.findByText("notes.txt")).toBeDefined();
    await expectPanelWidth(DEFAULT_ARTIFACT_PANEL_WIDTH);

    offsetWidthSpy.mockReturnValue(900);
    act(() => ResizeObserverMock.instances[0].trigger());

    await expectPanelWidth(900 - PANEL_RESERVED_WIDTH);
  });

  it("keeps the header Close button for standalone hosts", async () => {
    offsetWidthSpy = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(2000);

    render(<ArtifactPanel />);

    expect(await screen.findByText("notes.txt")).toBeDefined();
    expect(screen.getByRole("button", { name: "Close" })).toBeDefined();
  });

  it("drops the header Close button when the host closes it externally", async () => {
    offsetWidthSpy = vi
      .spyOn(HTMLElement.prototype, "offsetWidth", "get")
      .mockReturnValue(2000);

    render(<ArtifactPanel hasExternalClose />);

    expect(await screen.findByText("notes.txt")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Close" })).toBeNull();
  });
});
