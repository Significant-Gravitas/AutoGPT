import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ToolUIPart } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";
import { OPENUI_MIME } from "@/lib/openui/file";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";
import { registerChatSender } from "../../../helpers/sendToChat";
import { useCopilotUIStore } from "../../../store";

const router = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ ...router, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const SESSION = "card-session";
const source = formView(
  'Field("notes", "Notes", "Original", "", false)',
).replace("[form])", '[form], "saved-review")');
const draftPath = `*/api/chat/sessions/${SESSION}/ui-drafts/saved-review`;
const viewStatePath = "*/api/workspace/files/file-1/view-state";

function viewState(overrides: Record<string, unknown> = {}) {
  return {
    file_id: "file-1",
    view_id: "saved-review",
    source,
    checksum: "c1",
    state: {},
    revision: 2,
    origin_session_id: SESSION,
    ...overrides,
  };
}

function showCard(
  output: Record<string, unknown> = {},
  send = vi.fn().mockResolvedValue(undefined),
) {
  const part: ToolUIPart = {
    type: "tool-render_ui",
    toolCallId: "saved-ui",
    state: "output-available",
    input: {},
    output: {
      type: "ui_rendered",
      version: 1,
      session_id: SESSION,
      source,
      message: "A review form",
      ...output,
    },
  };
  render(
    <CopilotChatActionsProvider onSend={send} chatSurface="copilot">
      <ChainMessageParts
        parts={[part]}
        messageID="saved-message"
        isCurrentlyStreaming={false}
      />
    </CopilotChatActionsProvider>,
  );
  return send;
}

const savedOutput = { file_id: "file-1", file_name: "Review.openui" };

async function findNotes() {
  return (await screen.findByLabelText(
    "Notes",
    {},
    { timeout: 10000 },
  )) as HTMLInputElement;
}

describe("saved interactive views in chat", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
    router.push.mockClear();
  });
  afterEach(() => {
    useCopilotUIStore.getState().resetArtifactPanel();
  });

  it("saves an unsaved view to files and turns the card into a saved view", async () => {
    const saves: unknown[] = [];
    let saved = false;
    server.use(
      http.get(draftPath, () =>
        HttpResponse.json({
          revision: 0,
          source,
          state: {},
          tool_call_id: "saved-ui",
          writable: !saved,
          file_id: saved ? "file-1" : null,
        }),
      ),
      http.post(`${draftPath}/save`, async ({ request }) => {
        saves.push(await request.json());
        saved = true;
        return HttpResponse.json({
          file_id: "file-1",
          file_name: "Review.openui",
          file_path: `/sessions/${SESSION}/Review.openui`,
          file_uri: `workspace://file-1#${OPENUI_MIME}`,
          view_id: "saved-review",
          checksum: "c1",
        });
      }),
      http.get(viewStatePath, () => HttpResponse.json(viewState())),
    );
    showCard();
    fireEvent.click(
      await screen.findByRole(
        "button",
        { name: "Save to files" },
        { timeout: 10000 },
      ),
    );
    const dialog = await screen.findByRole("dialog");
    const name = within(dialog).getByLabelText("File name") as HTMLInputElement;
    expect(name.value).toBe("Review.openui");
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() => expect(saves).toEqual([{ filename: "Review.openui" }]));
    expect(await screen.findByText("Review.openui")).toBeDefined();
    expect(screen.getByRole("button", { name: "Open" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Save to files" })).toBeNull();
  });

  it("shows a 409 filename clash in the save dialog", async () => {
    server.use(
      http.get(draftPath, () =>
        HttpResponse.json({
          revision: 0,
          source,
          state: {},
          tool_call_id: "saved-ui",
          writable: true,
          file_id: null,
        }),
      ),
      http.post(`${draftPath}/save`, () =>
        HttpResponse.json({ detail: "exists" }, { status: 409 }),
      ),
    );
    showCard();
    fireEvent.click(
      await screen.findByRole(
        "button",
        { name: "Save to files" },
        { timeout: 10000 },
      ),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));
    expect(
      await within(dialog).findByText(/A file with this name already exists/),
    ).toBeDefined();
  });

  it("opens a saved view in the side panel from its card header", async () => {
    server.use(http.get(viewStatePath, () => HttpResponse.json(viewState())));
    showCard(savedOutput);
    await findNotes();
    expect(screen.getByText("Review.openui")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    const active = useCopilotUIStore.getState().artifactPanel.activeArtifact;
    expect(active?.id).toBe("file-1");
    expect(active?.mimeType).toBe(OPENUI_MIME);
    expect(active?.title).toBe("Review.openui");
  });

  it("saves edits to the file with its revision and checksum", async () => {
    const writes: Record<string, unknown>[] = [];
    server.use(
      http.get(viewStatePath, () => HttpResponse.json(viewState())),
      http.put(viewStatePath, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        writes.push(body);
        return HttpResponse.json(viewState({ revision: 3, state: body.state }));
      }),
    );
    showCard(savedOutput);
    fireEvent.change(await findNotes(), { target: { value: "Changed here" } });
    await waitFor(() => expect(writes.length).toBe(1), { timeout: 3000 });
    expect(writes[0].expected_revision).toBe(2);
    expect(writes[0].expected_checksum).toBe("c1");
    expect(JSON.stringify(writes[0].state)).toContain("Changed here");
  });

  it("keeps edits after a file conflict and reloads the saved edits on request", async () => {
    let conflict = false;
    server.use(
      http.get(viewStatePath, () =>
        HttpResponse.json(
          viewState({
            revision: conflict ? 4 : 2,
            state: conflict
              ? {
                  review: {
                    notes: { value: "Other device", componentType: "Field" },
                  },
                }
              : {},
          }),
        ),
      ),
      http.put(viewStatePath, () => {
        conflict = true;
        return HttpResponse.json({ detail: "Conflict" }, { status: 409 });
      }),
    );
    showCard(savedOutput);
    fireEvent.change(await findNotes(), { target: { value: "My edits" } });
    const reload = await screen.findByRole(
      "button",
      { name: "Reload saved draft" },
      { timeout: 3000 },
    );
    expect(screen.getByText("Not synced", { exact: false })).toBeDefined();
    expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
      "My edits",
    );
    fireEvent.click(reload);
    await waitFor(() =>
      expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
        "Other device",
      ),
    );
  });

  it("submits in this chat when the view belongs to it", async () => {
    server.use(http.get(viewStatePath, () => HttpResponse.json(viewState())));
    const send = showCard(savedOutput);
    await findNotes();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain("Use my choices");
    expect(router.push).not.toHaveBeenCalled();
  });

  it("sends a submit from another chat to the view's home chat and opens it", async () => {
    server.use(
      http.get(viewStatePath, () =>
        HttpResponse.json(viewState({ origin_session_id: "home-chat" })),
      ),
    );
    const send = showCard(savedOutput);
    await findNotes();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() =>
      expect(router.push).toHaveBeenCalledWith("/copilot?sessionId=home-chat"),
    );
    expect(send).not.toHaveBeenCalled();

    const homeSend = vi.fn();
    const unregister = registerChatSender("home-chat", homeSend);
    expect(homeSend).toHaveBeenCalledTimes(1);
    expect(homeSend.mock.calls[0][0]).toContain("Use my choices");
    expect(homeSend.mock.calls[0][0]).toContain(
      "From the interactive view: Review",
    );
    unregister();
  });

  it("keeps edits but turns off submit when the home chat is gone", async () => {
    server.use(
      http.get(viewStatePath, () =>
        HttpResponse.json(viewState({ origin_session_id: null })),
      ),
    );
    const send = showCard(savedOutput);
    const notes = await findNotes();
    expect(notes.disabled).toBe(false);
    const submit = screen.getByRole("button", { name: "Continue" });
    expect(submit.hasAttribute("disabled")).toBe(true);
    expect(
      screen.getByText("The chat this view came from no longer exists."),
    ).toBeDefined();
    fireEvent.click(submit);
    expect(send).not.toHaveBeenCalled();
  });

  it("keeps an older revision of a saved view read-only", async () => {
    server.use(
      http.get(viewStatePath, () =>
        HttpResponse.json(
          viewState({ source: source.replace("Original", "Newer") }),
        ),
      ),
    );
    showCard(savedOutput);
    expect((await findNotes()).disabled).toBe(true);
    expect(
      screen.getByText(/A newer version of this view is saved in your files/),
    ).toBeDefined();
  });
});
