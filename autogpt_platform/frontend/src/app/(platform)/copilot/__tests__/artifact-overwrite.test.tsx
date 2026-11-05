import { globalRegistry } from "@/components/contextual/OutputRenderers";
import { server } from "@/mocks/mock-server";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearContentCache,
  getCachedArtifactContent,
} from "../components/ArtifactPanel/components/useArtifactContent";
import { CopilotPage } from "../CopilotPage";
import { useCopilotUIStore } from "../store";
import {
  copilotStreamSequenceHandler,
  renderHost,
  resetChatRuntimes,
  STREAM_PATHS,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("../useIsMobile", () => ({ useIsMobile: () => false }));

const streamPath = vi.hoisted(() => ({ runtime: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "copilot-stream-runtime" ? streamPath.runtime : false,
  };
});

const V1_ID = "11111111-1111-4111-8111-111111111111";
const V2_ID = "22222222-2222-4222-8222-222222222222";
const V3_ID = "44444444-4444-4444-8444-444444444444";
const V1_TEXT = "Version one of the report";
const V2_TEXT = "Version two of the report";
const V3_TEXT = "Version three of the report";

const HISTORY = [
  { role: "user", content: "Write me a report", sequence: 1 },
  {
    role: "assistant",
    content: `Here it is: [report.md](workspace://${V1_ID}#text/markdown)`,
    sequence: 2,
  },
];

function writeChunks(fileId: string, replacedFileId: string | null) {
  const toolCallId = `write-${fileId}`;
  return [
    {
      type: "tool-input-available",
      toolCallId,
      toolName: "write_workspace_file",
      input: { filename: "report.md", content: "text", overwrite: true },
    },
    {
      type: "tool-output-available",
      toolCallId,
      output: JSON.stringify({
        type: "workspace_file_written",
        message: "Wrote report.md to workspace (25 bytes)",
        file_id: fileId,
        name: "report.md",
        path: "/report.md",
        mime_type: "text/markdown",
        size_bytes: 25,
        download_url: `workspace://${fileId}#text/markdown`,
        ...(replacedFileId ? { replaced_file_id: replacedFileId } : {}),
      }),
    },
  ] satisfies UIMessageChunk[];
}

function turnWithWrites(...writes: UIMessageChunk[][]): UIMessageChunk[] {
  return [
    { type: "start", messageId: "turn-2" },
    { type: "start-step" },
    ...writes.flat(),
    { type: "text-start", id: "answer" },
    { type: "text-delta", id: "answer", delta: "Report written." },
    { type: "text-end", id: "answer" },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

function overwriteTurn(replacedFileId: string | null) {
  return turnWithWrites(writeChunks(V2_ID, replacedFileId));
}

function deferredText() {
  let release = (_text: string) => {};
  const text = new Promise<string>((resolve) => {
    release = resolve;
  });
  return { text, release };
}

// Downloads a test holds open, settled after each test so nothing is left
// pending when the file ends.
const heldDownloads: Array<() => void> = [];

function holdDownload() {
  const download = deferredText();
  heldDownloads.push(() => download.release(""));
  return download;
}

function fileDownloads(
  texts: { v2?: Promise<string> | string; v3?: Promise<string> | string } = {},
) {
  const fetched: string[] = [];
  server.use(
    http.get("*/api/workspace/files/:id/download", async ({ params }) => {
      const id = String(params.id);
      fetched.push(id);
      if (id === V1_ID) return HttpResponse.text(V1_TEXT);
      if (id === V2_ID) return HttpResponse.text(await (texts.v2 ?? V2_TEXT));
      if (id === V3_ID) return HttpResponse.text(await (texts.v3 ?? V3_TEXT));
      return new HttpResponse(null, { status: 404 });
    }),
    http.get("*/api/workspace/files", () =>
      HttpResponse.json({ files: [], offset: 0, has_more: false }),
    ),
  );
  return fetched;
}

function streamReturns(...turns: UIMessageChunk[][]) {
  server.use(
    copilotStreamSequenceHandler({
      baseUrl: TEST_BACKEND_BASE_URL,
      sessionId: TEST_SESSION_ID,
      chunksPerTurn: turns,
    }),
  );
}

function renderChatWithReport() {
  return renderHost({
    ui: <CopilotPage />,
    sessionOverride: { messages: HISTORY },
  });
}

function panel() {
  return document.querySelector<HTMLElement>("[data-artifact-panel]");
}

function openFileId() {
  return useCopilotUIStore.getState().artifactPanel.activeArtifact?.id;
}

function scrollPage(text: string) {
  return within(panel()!)
    .getByText(text)
    .closest<HTMLElement>(".overflow-y-auto")!;
}

async function openReport() {
  const user = userEvent.setup();
  const card = await screen.findByRole("button", { name: /report\.md/i });
  await user.click(card);
  await waitFor(() => {
    expect(panel()).not.toBeNull();
    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();
  });
}

async function waitForTurnEnd() {
  await screen.findByText(/report written/i);
  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Stop" })).toBeNull(),
  );
}

beforeEach(() => {
  resetChatRuntimes();
  clearContentCache();
  useCopilotUIStore.getState().resetArtifactPanel();
  useCopilotUIStore.getState().closeArtifactPanel({ persist: false });
});

afterEach(() => {
  heldDownloads.splice(0).forEach((settle) => settle());
  vi.restoreAllMocks();
  resetChatRuntimes();
});

describe.each(STREAM_PATHS)("overwriting a file on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  it("moves the open panel to the new version with no click and no error", async () => {
    const fetched = fileDownloads();
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();
    await openReport();

    await typeAndSend("Rewrite the report");

    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(within(panel()!).queryByText(V1_TEXT)).toBeNull();
    expect(within(panel()!).queryByRole("alert")).toBeNull();
    expect(fetched.filter((id) => id === V1_ID)).toHaveLength(1);
    expect(getCachedArtifactContent(V1_ID)).toBeUndefined();
    // The new version is not somewhere to go back from.
    expect(useCopilotUIStore.getState().artifactPanel.history).toEqual([]);
  });

  it("keeps the text on screen and the scroll position until the new text lands", async () => {
    const v2 = holdDownload();
    fileDownloads({ v2: v2.text });
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();
    await openReport();
    const page = scrollPage(V1_TEXT);
    page.scrollTop = 240;

    await typeAndSend("Rewrite the report");
    await waitFor(() => expect(openFileId()).toBe(V2_ID));
    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();

    v2.release(V2_TEXT);
    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(panel()!.contains(page)).toBe(true);
    expect(page.scrollTop).toBe(240);
  });

  it("holds the text through a second overwrite while the first download is pending", async () => {
    const v2 = holdDownload();
    const v3 = holdDownload();
    fileDownloads({ v2: v2.text, v3: v3.text });
    streamReturns(
      overwriteTurn(V1_ID),
      turnWithWrites(writeChunks(V3_ID, V2_ID)),
    );
    renderChatWithReport();
    await openReport();
    const page = scrollPage(V1_TEXT);
    page.scrollTop = 240;

    await typeAndSend("Rewrite the report");
    await waitFor(() => expect(openFileId()).toBe(V2_ID));
    await waitForTurnEnd();
    await typeAndSend("Once more");
    await waitFor(() => expect(openFileId()).toBe(V3_ID));
    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();

    v3.release(V3_TEXT);
    await waitFor(() =>
      expect(within(panel()!).getByText(V3_TEXT)).toBeDefined(),
    );
    expect(panel()!.contains(page)).toBe(true);
    expect(page.scrollTop).toBe(240);
  });

  it("holds the new text the same way when Otto overwrites it again", async () => {
    const v3 = holdDownload();
    fileDownloads({ v3: v3.text });
    streamReturns(
      overwriteTurn(V1_ID),
      turnWithWrites(writeChunks(V3_ID, V2_ID)),
    );
    renderChatWithReport();
    await openReport();

    await typeAndSend("Rewrite the report");
    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    await waitForTurnEnd();
    const page = scrollPage(V2_TEXT);
    page.scrollTop = 240;

    await typeAndSend("Once more");
    await waitFor(() => expect(openFileId()).toBe(V3_ID));
    expect(within(panel()!).getByText(V2_TEXT)).toBeDefined();

    v3.release(V3_TEXT);
    await waitFor(() =>
      expect(within(panel()!).getByText(V3_TEXT)).toBeDefined(),
    );
    expect(panel()!.contains(page)).toBe(true);
    expect(page.scrollTop).toBe(240);
  });

  it("doesn't refetch or drop the text when the open version's card is clicked mid-download", async () => {
    const v2 = holdDownload();
    const fetched = fileDownloads({ v2: v2.text });
    streamReturns(
      turnWithWrites(writeChunks(V2_ID, V1_ID), [
        { type: "text-start", id: "link" },
        {
          type: "text-delta",
          id: "link",
          delta: `Updated: [new-report.md](workspace://${V2_ID}#text/markdown)`,
        },
        { type: "text-end", id: "link" },
      ]),
    );
    renderChatWithReport();
    await openReport();

    await typeAndSend("Rewrite the report");
    await waitFor(() => expect(fetched).toContain(V2_ID));
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: /new-report\.md/i }));

    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();
    v2.release(V2_TEXT);
    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(fetched.filter((id) => id === V2_ID)).toHaveLength(1);
  });

  it("holds the place and the source view when one turn writes the file twice", async () => {
    fileDownloads();
    streamReturns(
      turnWithWrites(writeChunks(V2_ID, V1_ID), writeChunks(V3_ID, V2_ID)),
    );
    renderChatWithReport();
    await openReport();
    await userEvent
      .setup()
      .click(within(panel()!).getByRole("button", { name: "Source" }));
    const page = scrollPage(V1_TEXT);
    page.scrollTop = 240;

    await typeAndSend("Rewrite the report twice");

    await waitFor(() =>
      expect(within(panel()!).getByText(V3_TEXT)).toBeDefined(),
    );
    expect(panel()!.contains(page)).toBe(true);
    expect(page.scrollTop).toBe(240);
    expect(
      within(panel()!)
        .getByRole("button", { name: "Source" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("keeps the source view on when the open version's card is clicked", async () => {
    fileDownloads();
    streamReturns(
      turnWithWrites(writeChunks(V2_ID, V1_ID), [
        { type: "text-start", id: "link" },
        {
          type: "text-delta",
          id: "link",
          delta: `Updated: [new-report.md](workspace://${V2_ID}#text/markdown)`,
        },
        { type: "text-end", id: "link" },
      ]),
    );
    renderChatWithReport();
    await openReport();
    const user = userEvent.setup();
    await user.click(within(panel()!).getByRole("button", { name: "Source" }));

    await typeAndSend("Rewrite the report");
    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    await user.click(
      await screen.findByRole("button", { name: /new-report\.md/i }),
    );

    expect(openFileId()).toBe(V2_ID);
    expect(
      within(panel()!)
        .getByRole("button", { name: "Source" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("recovers when the old version crashed the preview and the new one renders", async () => {
    const getRenderer = globalRegistry.getRenderer.bind(globalRegistry);
    vi.spyOn(globalRegistry, "getRenderer").mockImplementation(
      (content, metadata) => {
        if (content === V1_TEXT) throw new Error("Cannot render version one");
        return getRenderer(content, metadata);
      },
    );
    fileDownloads();
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: /report\.md/i }),
    );
    await waitFor(() =>
      expect(within(panel()!).getByRole("alert").textContent).toContain(
        "Cannot render version one",
      ),
    );

    await typeAndSend("Fix the report");

    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(within(panel()!).queryByRole("alert")).toBeNull();
  });

  it("stays on the source view", async () => {
    fileDownloads();
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();
    await openReport();
    await userEvent
      .setup()
      .click(within(panel()!).getByRole("button", { name: "Source" }));

    await typeAndSend("Rewrite the report");

    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(
      within(panel()!)
        .getByRole("button", { name: "Source" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("leaves a closed panel closed", async () => {
    const fetched = fileDownloads();
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();

    await typeAndSend("Rewrite the report");

    await waitForTurnEnd();
    expect(panel()).toBeNull();
    expect(fetched).toEqual([]);
  });

  it("stays on the open file when a different file is overwritten", async () => {
    const fetched = fileDownloads();
    streamReturns(overwriteTurn("33333333-3333-4333-8333-333333333333"));
    renderChatWithReport();
    await openReport();

    await typeAndSend("Rewrite the other file");

    await waitForTurnEnd();
    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();
    expect(fetched).not.toContain(V2_ID);
  });
});
