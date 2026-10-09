import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MAX_ATTACHMENTS } from "../../../helpers/workspaceAttachments";
import { ChatInput } from "../ChatInput";

// Only the composer's periphery is stubbed: FileChips, ComposerPlusMenu and
// the cap notice stay real, because the cap is a claim about what the user
// ends up seeing and being able to click.
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2ListChatConnections: () => ({
    data: { status: 200, data: { offers: [] } },
    isLoading: false,
    isPending: false,
    isError: false,
  }),
}));

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    copilotLlmModel: "standard",
    setCopilotLlmModel: vi.fn(),
    copilotLlmAuth: { authProvider: "platform", credentialId: null },
    setCopilotLlmAuth: vi.fn(),
    isDryRun: false,
    setIsDryRun: vi.fn(),
    initialPrompt: null,
    setInitialPrompt: vi.fn(),
    sentMessageCount: 0,
    notifyMessageSent: vi.fn(),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    CHAT_WORKSPACE_FILES: "chat-workspace-files",
  },
  useGetFlag: () => false,
}));

vi.mock("../useVoiceRecording", () => ({
  useVoiceRecording: () => ({
    isRecording: false,
    isTranscribing: false,
    transcriptionError: null,
    hasFailedRecording: false,
    retryTranscription: vi.fn(),
    downloadFailedRecording: vi.fn(),
    dismissTranscriptionError: vi.fn(),
    elapsedTime: 0,
    toggleRecording: vi.fn(),
    handleKeyDown: vi.fn(),
    showMicButton: false,
    isInputDisabled: false,
    audioStream: null,
  }),
}));

vi.mock("@/components/ai-elements/prompt-input", () => ({
  PromptInputSubmit: ({ disabled }: { disabled?: boolean }) => (
    <button disabled={disabled} data-testid="submit">
      Send
    </button>
  ),
  PromptInputButton: () => null,
  PromptInputTextarea: (props: {
    id?: string;
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
    onPaste?: React.ClipboardEventHandler<HTMLTextAreaElement>;
  }) => (
    <textarea
      id={props.id}
      value={props.value}
      onChange={props.onChange}
      onPaste={props.onPaste}
      data-testid="textarea"
    />
  ),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn() }),
}));

afterEach(() => {
  vi.clearAllMocks();
});

const OVER = 3;

describe("ChatInput attachment cap — pasting", () => {
  it(`keeps the first ${MAX_ATTACHMENTS} and names how many it refused`, async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);

    pasteFiles(makeFiles(MAX_ATTACHMENTS + OVER));

    const notice = await screen.findByRole("status");
    expect(notice.textContent).toContain(
      `Up to ${MAX_ATTACHMENTS} attachments per message — ${OVER} not added`,
    );
    expect(removeButtons()).toHaveLength(MAX_ATTACHMENTS);

    fireEvent.submit(textarea().closest("form")!);
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    const [, files] = onSend.mock.calls[0];
    expect(files).toHaveLength(MAX_ATTACHMENTS);
    expect(files.map((f: File) => f.name)).toEqual(
      makeFiles(MAX_ATTACHMENTS).map((f) => f.name),
    );
  });

  it("leaves the typed message untouched when it refuses files", async () => {
    render(<ChatInput onSend={vi.fn()} />);
    fireEvent.change(textarea(), {
      target: { value: "here are my quarterly notes" },
    });

    pasteFiles(makeFiles(MAX_ATTACHMENTS + OVER));

    await screen.findByRole("status");
    expect(textarea().value).toBe("here are my quarterly notes");
  });

  it("counts what is already attached, refusing only the overflow", async () => {
    render(<ChatInput onSend={vi.fn()} />);

    pasteFiles(makeFiles(MAX_ATTACHMENTS - 1));
    expect(screen.queryByRole("status")).toBeNull();

    pasteFiles(makeFiles(3, "late"));

    const notice = await screen.findByRole("status");
    expect(notice.textContent).toContain("2 not added");
    expect(removeButtons()).toHaveLength(MAX_ATTACHMENTS);
  });
});

describe("ChatInput attachment cap — dropping", () => {
  it(`keeps the first ${MAX_ATTACHMENTS} of a drop and names the refusal`, async () => {
    render(
      <ChatInput
        onSend={vi.fn()}
        droppedFiles={makeFiles(MAX_ATTACHMENTS + OVER)}
        onDroppedFilesConsumed={vi.fn()}
      />,
    );

    const notice = await screen.findByRole("status");
    expect(notice.textContent).toContain(`${OVER} not added`);
    expect(removeButtons()).toHaveLength(MAX_ATTACHMENTS);
  });

  it("refuses a drop that lands on an already-full composer", async () => {
    const { rerender } = render(
      <ChatInput onSend={vi.fn()} onDroppedFilesConsumed={vi.fn()} />,
    );
    pasteFiles(makeFiles(MAX_ATTACHMENTS));
    await waitFor(() => expect(removeButtons()).toHaveLength(MAX_ATTACHMENTS));

    rerender(
      <ChatInput
        onSend={vi.fn()}
        droppedFiles={makeFiles(2, "dropped")}
        onDroppedFilesConsumed={vi.fn()}
      />,
    );

    const notice = await screen.findByRole("status");
    expect(notice.textContent).toContain("2 not added");
    expect(removeButtons()).toHaveLength(MAX_ATTACHMENTS);
  });
});

describe("ChatInput attachment cap — the notice and the affordance", () => {
  it("dismisses the notice on request", async () => {
    render(<ChatInput onSend={vi.fn()} />);
    pasteFiles(makeFiles(MAX_ATTACHMENTS + OVER));
    await screen.findByRole("status");

    fireEvent.click(
      screen.getByRole("button", { name: /dismiss attachment limit/i }),
    );

    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  });

  it("clears the notice once a removal frees a slot", async () => {
    render(<ChatInput onSend={vi.fn()} />);
    pasteFiles(makeFiles(MAX_ATTACHMENTS + OVER));
    await screen.findByRole("status");

    fireEvent.click(removeButtons()[0]);

    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  });

  it("disables the attach affordance at the cap and re-enables it below", async () => {
    render(<ChatInput onSend={vi.fn()} />);
    pasteFiles(makeFiles(MAX_ATTACHMENTS));

    fireEvent.pointerDown(screen.getByTestId("composer-plus-button"), {
      button: 0,
    });
    const attach = await screen.findByRole("menuitem", {
      name: /attach file/i,
    });
    expect(attach.getAttribute("data-disabled")).not.toBeNull();
    expect(attach.textContent).toContain(`${MAX_ATTACHMENTS} max`);

    fireEvent.keyDown(attach, { key: "Escape" });
    fireEvent.click(removeButtons()[0]);
    fireEvent.pointerDown(screen.getByTestId("composer-plus-button"), {
      button: 0,
    });

    await waitFor(() =>
      expect(
        screen
          .getByRole("menuitem", { name: /attach file/i })
          .getAttribute("data-disabled"),
      ).toBeNull(),
    );
  });
});

function textarea() {
  return screen.getByTestId("textarea") as HTMLTextAreaElement;
}

function removeButtons() {
  return screen.queryAllByRole("button", { name: /^Remove / });
}

function pasteFiles(files: File[]) {
  fireEvent.paste(textarea(), { clipboardData: { files } });
}

function makeFiles(count: number, prefix = "doc") {
  return Array.from(
    { length: count },
    (_, i) =>
      new File([`bytes-${i}`], `${prefix}-${i}.pdf`, {
        type: "application/pdf",
      }),
  );
}
