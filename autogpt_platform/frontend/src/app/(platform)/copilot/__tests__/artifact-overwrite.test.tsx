import { server } from "@/mocks/mock-server";
import { copilotStreamHandler } from "@/tests/integrations/copilot-sse";
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
const V1_TEXT = "Version one of the report";
const V2_TEXT = "Version two of the report";

const HISTORY = [
  { role: "user", content: "Write me a report", sequence: 1 },
  {
    role: "assistant",
    content: `Here it is: [report.md](workspace://${V1_ID}#text/markdown)`,
    sequence: 2,
  },
];

function overwriteTurn(replacedFileId: string | null): UIMessageChunk[] {
  return [
    { type: "start", messageId: "turn-2" },
    { type: "start-step" },
    {
      type: "tool-input-available",
      toolCallId: "write-1",
      toolName: "write_workspace_file",
      input: { filename: "report.md", content: V2_TEXT, overwrite: true },
    },
    {
      type: "tool-output-available",
      toolCallId: "write-1",
      output: JSON.stringify({
        type: "workspace_file_written",
        message: "Wrote report.md to workspace (25 bytes)",
        file_id: V2_ID,
        name: "report.md",
        path: "/report.md",
        mime_type: "text/markdown",
        size_bytes: 25,
        download_url: `workspace://${V2_ID}#text/markdown`,
        ...(replacedFileId ? { replaced_file_id: replacedFileId } : {}),
      }),
    },
    { type: "text-start", id: "answer" },
    { type: "text-delta", id: "answer", delta: "Report written." },
    { type: "text-end", id: "answer" },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

// The stored v1 is gone once Otto overwrites it, so a fetch of its id after
// the swap would show an error rather than old text.
function fileDownloads(v2Text: () => Promise<string> | string = () => V2_TEXT) {
  const fetched: string[] = [];
  server.use(
    http.get("*/api/workspace/files/:id/download", async ({ params }) => {
      const id = String(params.id);
      fetched.push(id);
      if (id === V1_ID) return HttpResponse.text(V1_TEXT);
      if (id === V2_ID) return HttpResponse.text(await v2Text());
      return new HttpResponse(null, { status: 404 });
    }),
    http.get("*/api/workspace/files", () =>
      HttpResponse.json({ files: [], offset: 0, has_more: false }),
    ),
  );
  return fetched;
}

function streamReturns(chunks: UIMessageChunk[]) {
  server.use(
    copilotStreamHandler({
      baseUrl: TEST_BACKEND_BASE_URL,
      sessionId: TEST_SESSION_ID,
      chunks,
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
    let releaseV2 = (_text: string) => {};
    fileDownloads(
      () =>
        new Promise<string>((resolve) => {
          releaseV2 = resolve;
        }),
    );
    streamReturns(overwriteTurn(V1_ID));
    renderChatWithReport();
    await openReport();

    const page = within(panel()!)
      .getByText(V1_TEXT)
      .closest<HTMLElement>(".overflow-y-auto")!;
    page.scrollTop = 240;

    await typeAndSend("Rewrite the report");
    await waitFor(() =>
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe(V2_ID),
    );
    expect(within(panel()!).getByText(V1_TEXT)).toBeDefined();

    releaseV2(V2_TEXT);
    await waitFor(() =>
      expect(within(panel()!).getByText(V2_TEXT)).toBeDefined(),
    );
    expect(panel()!.contains(page)).toBe(true);
    expect(page.scrollTop).toBe(240);
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
