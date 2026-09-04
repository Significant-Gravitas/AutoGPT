import {
  render,
  screen,
  fireEvent,
  act,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  NEW_SCHEDULED_TASK_PROMPT,
  NEW_SKILL_PROMPT,
} from "@/components/contextual/guidedPrompts";
import type { UIMessage } from "ai";
import { useRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../ChatInput";
import { useCopilotStop } from "../../../useCopilotStop";
import { toast } from "@/components/molecules/Toast/use-toast";

const mockCancel =
  vi.fn<(sessionId: string) => Promise<{ status: number; data: unknown }>>();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: (sessionId: string) => mockCancel(sessionId),
  // A lone platform connection whose tiers resolve to one model: the picker
  // presents nothing, which keeps these tests about the composer itself.
  useGetV2ListChatConnections: () => ({
    data: {
      status: 200,
      data: {
        offers: [
          {
            offer_id: "platform:deployment",
            provider_family: "autogpt",
            display_name: "AutoGPT Platform",
            auth_method: "deployment",
            credential_id: null,
            backed_by_label: "Your AutoGPT plan",
            description: "New chats are backed by your AutoGPT plan.",
            state: "ready",
            selectable: true,
            is_default: true,
            tiers: [
              {
                tier: "standard",
                label: "Balanced",
                selectable: true,
                display_model: "one-model",
              },
              {
                tier: "advanced",
                label: "Advanced",
                selectable: true,
                display_model: "one-model",
              },
            ],
            limitations: [],
          },
        ],
      },
    },
    isLoading: false,
    isPending: false,
    isError: false,
  }),
}));

let mockCopilotLlmModel = "standard";
const mockSetCopilotLlmModel = vi.fn((model: string) => {
  mockCopilotLlmModel = model;
});

let mockCopilotLlmAuthProvider = "platform";

let mockInitialPrompt: string | null = null;
const mockSetInitialPrompt = vi.fn((value: string | null) => {
  mockInitialPrompt = value;
});

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    copilotLlmModel: mockCopilotLlmModel,
    setCopilotLlmModel: mockSetCopilotLlmModel,
    copilotLlmAuth: {
      authProvider: mockCopilotLlmAuthProvider,
      credentialId: null,
    },
    setCopilotLlmAuth: vi.fn(),
    isDryRun: false,
    setIsDryRun: vi.fn(),
    initialPrompt: mockInitialPrompt,
    setInitialPrompt: mockSetInitialPrompt,
    sentMessageCount: 0,
    notifyMessageSent: vi.fn(),
  }),
}));

let mockFlagValue = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CHAT_MODE_OPTION: "CHAT_MODE_OPTION" },
  useGetFlag: () => mockFlagValue,
}));

// Off by default so the rest of the suite sees the production-build behaviour.
let mockTokenDevtoolEnabled = false;
vi.mock("../../../tokenDevtool/gate", () => ({
  isTokenDevtoolEnabled: () => mockTokenDevtoolEnabled,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn() }),
}));

vi.mock("../useVoiceRecording", () => ({
  useVoiceRecording: () => ({
    isRecording: false,
    isTranscribing: false,
    elapsedTime: 0,
    toggleRecording: vi.fn(),
    handleKeyDown: vi.fn(),
    showMicButton: false,
    isInputDisabled: false,
    audioStream: null,
  }),
}));

vi.mock("@/components/ai-elements/prompt-input", () => ({
  PromptInputSubmit: ({
    disabled,
    status,
    onStop,
  }: {
    disabled?: boolean;
    status?: string;
    onStop?: () => void;
  }) =>
    status === "streaming" ? (
      <button
        type="button"
        onClick={onStop}
        data-testid="stop"
        aria-label="Stop"
      >
        Stop
      </button>
    ) : (
      <button disabled={disabled} data-testid="submit">
        Send
      </button>
    ),
  PromptInputTextarea: function PromptInputTextarea(props: {
    id?: string;
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
    onPaste?: React.ClipboardEventHandler<HTMLTextAreaElement>;
    disabled?: boolean;
    placeholder?: string;
  }) {
    return (
      <textarea
        id={props.id}
        value={props.value}
        onChange={props.onChange}
        onPaste={props.onPaste}
        disabled={props.disabled}
        placeholder={props.placeholder}
        data-testid="textarea"
      />
    );
  },
  PromptInputButton: ({
    children,
    onClick,
    "aria-label": ariaLabel,
  }: {
    children?: React.ReactNode;
    onClick?: React.MouseEventHandler<HTMLButtonElement>;
    "aria-label"?: string;
  }) => (
    <button aria-label={ariaLabel} onClick={onClick} data-testid="queue-btn">
      {children}
    </button>
  ),
}));

// InputGroup/InputGroupAddon render as-is: they are plain presentational
// wrappers, and stubbing them hid the addon contract (role, data-align,
// click-to-focus) that the composer relies on.

vi.mock("../components/ComposerPlusMenu", () => ({
  ComposerPlusMenu: ({
    onClearGuidedPrompt,
  }: {
    onClearGuidedPrompt?: () => void;
  }) => (
    <button
      type="button"
      data-testid="attachment-menu"
      onClick={() => onClearGuidedPrompt?.()}
    />
  ),
}));
vi.mock("../components/FileChips", () => ({
  FileChips: () => null,
}));
vi.mock("../components/RecordingButton", () => ({
  RecordingButton: () => null,
}));
vi.mock("../components/RecordingIndicator", () => ({
  RecordingIndicator: () => null,
}));
vi.mock("../components/DryRunToggleButton", () => ({
  DryRunToggleButton: ({
    onToggle,
  }: {
    isDryRun: boolean;
    isStreaming: boolean;
    readOnly: boolean;
    onToggle: () => void;
  }) => (
    <button data-testid="dry-run-toggle" onClick={onToggle}>
      Dry Run
    </button>
  ),
}));

const mockOnSend = vi.fn();

afterEach(() => {
  vi.clearAllMocks();
  mockCancel.mockReset();
  mockCopilotLlmModel = "standard";
  mockCopilotLlmAuthProvider = "platform";
  mockFlagValue = false;
  mockTokenDevtoolEnabled = false;
  mockInitialPrompt = null;
});

describe("ChatInput composer row", () => {
  it("keeps the leading and trailing addons on their declared sides", () => {
    render(<ChatInput onSend={mockOnSend} sessionId="session-1" />);

    const aligns = Array.from(
      document.querySelectorAll("[data-slot=input-group-addon]"),
    ).map((addon) => addon.getAttribute("data-align"));

    expect(aligns).toEqual(["inline-start", "inline-end"]);
  });

  it("focuses the message box when the addon gutter is clicked", () => {
    render(<ChatInput onSend={mockOnSend} sessionId="session-1" />);

    // The composer autofocuses on mount, so blur first — otherwise the
    // assertion below would pass without the addon doing anything.
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    textarea.blur();
    expect(document.activeElement).not.toBe(textarea);

    fireEvent.click(
      document.querySelector("[data-slot=input-group-addon]") as Element,
    );

    expect(document.activeElement).toBe(textarea);
  });
});

describe("ChatInput token devtool badge", () => {
  it("renders the badge while the brain-dump tray is disabled", () => {
    // The tray is brain-dump-only; the badge must not depend on that flag.
    mockFlagValue = false;
    mockTokenDevtoolEnabled = true;
    render(<ChatInput onSend={mockOnSend} sessionId="session-1" />);

    expect(screen.getByRole("button", { name: /Token devtool/ })).toBeDefined();
  });

  it("renders the badge inside the tray when brain dump is enabled", () => {
    mockFlagValue = true;
    mockTokenDevtoolEnabled = true;
    render(<ChatInput onSend={mockOnSend} sessionId="session-1" />);

    expect(screen.getByRole("button", { name: /Token devtool/ })).toBeDefined();
  });

  it("stays hidden when the devtool gate is off", () => {
    mockTokenDevtoolEnabled = false;
    render(<ChatInput onSend={mockOnSend} sessionId="session-1" />);

    expect(screen.queryByRole("button", { name: /Token devtool/ })).toBeNull();
  });

  it("stays hidden before a session exists", () => {
    mockTokenDevtoolEnabled = true;
    render(<ChatInput onSend={mockOnSend} sessionId={null} />);

    expect(screen.queryByRole("button", { name: /Token devtool/ })).toBeNull();
  });
});

describe("ChatInput Codex route", () => {
  it("keeps Claude SDK file attachments available for the Codex route", () => {
    mockCopilotLlmAuthProvider = "codex";
    render(<ChatInput onSend={mockOnSend} />);

    expect(screen.getByTestId("attachment-menu")).toBeTruthy();
  });

  it("does not report empty dropped files as consumed for the Codex route", () => {
    mockCopilotLlmAuthProvider = "codex";
    const onDroppedFilesConsumed = vi.fn();
    render(
      <ChatInput
        onSend={mockOnSend}
        droppedFiles={[]}
        onDroppedFilesConsumed={onDroppedFilesConsumed}
      />,
    );

    expect(onDroppedFilesConsumed).not.toHaveBeenCalled();
  });

  it("hides the route selector when only one subsidized transport is connected", () => {
    mockFlagValue = true;
    const { rerender } = render(<ChatInput onSend={mockOnSend} />);
    expect(screen.queryByLabelText(/AI connection:/i)).toBeNull();

    rerender(<ChatInput onSend={mockOnSend} hasSession />);
    expect(screen.queryByLabelText(/AI connection:/i)).toBeNull();
  });
});

describe("ChatInput queue button", () => {
  it("does not render queue button when not streaming", () => {
    render(<ChatInput onSend={mockOnSend} onEnqueue={vi.fn()} />);
    expect(screen.queryByLabelText(/queue message/i)).toBeNull();
  });

  it("does not render queue button when streaming but no text typed", () => {
    render(<ChatInput onSend={mockOnSend} onEnqueue={vi.fn()} isStreaming />);
    expect(screen.queryByLabelText(/queue message/i)).toBeNull();
  });

  it("renders queue button when streaming with text and onEnqueue provided", () => {
    render(<ChatInput onSend={mockOnSend} onEnqueue={vi.fn()} isStreaming />);
    const textarea = screen.getByTestId("textarea");
    fireEvent.change(textarea, { target: { value: "follow-up question" } });
    expect(screen.getByLabelText(/queue message/i)).toBeDefined();
  });

  it("calls onEnqueue with trimmed text when queue button clicked", async () => {
    const mockOnEnqueue = vi.fn().mockResolvedValue(undefined);
    render(
      <ChatInput onSend={mockOnSend} onEnqueue={mockOnEnqueue} isStreaming />,
    );
    const textarea = screen.getByTestId("textarea");
    fireEvent.change(textarea, { target: { value: "  hello  " } });
    await act(async () => {
      fireEvent.click(screen.getByLabelText(/queue message/i));
    });
    expect(mockOnEnqueue).toHaveBeenCalledWith("hello");
  });

  it("clears textarea after successful enqueue", async () => {
    const mockOnEnqueue = vi.fn().mockResolvedValue(undefined);
    render(
      <ChatInput onSend={mockOnSend} onEnqueue={mockOnEnqueue} isStreaming />,
    );
    const textarea = screen.getByTestId("textarea");
    fireEvent.change(textarea, { target: { value: "my message" } });
    fireEvent.click(screen.getByLabelText(/queue message/i));
    await waitFor(() => {
      expect((textarea as HTMLTextAreaElement).value).toBe("");
    });
  });

  it("preserves textarea text if queue button clicked with empty input", async () => {
    const mockOnEnqueue = vi.fn().mockResolvedValue(undefined);
    render(
      <ChatInput onSend={mockOnSend} onEnqueue={mockOnEnqueue} isStreaming />,
    );
    const textarea = screen.getByTestId("textarea");
    // No text typed — queue button should not render
    expect(screen.queryByLabelText(/queue message/i)).toBeNull();
    // onEnqueue must not be called
    expect(mockOnEnqueue).not.toHaveBeenCalled();
    // textarea stays empty
    expect((textarea as HTMLTextAreaElement).value).toBe("");
  });
});

describe("ChatInput dry-run toggle", () => {
  it("does not render dry-run toggle when flag is disabled", () => {
    mockFlagValue = false;
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.queryByTestId("dry-run-toggle")).toBeNull();
  });

  it("renders dry-run toggle when flag is enabled and no session", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} hasSession={false} />);
    expect(screen.getByTestId("dry-run-toggle")).toBeDefined();
  });

  it("hides dry-run toggle when session is active and isDryRun is false", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} hasSession />);
    // isDryRun is false in mock, hasSession is true → toggle hidden
    expect(screen.queryByTestId("dry-run-toggle")).toBeNull();
  });

  it("calls setIsDryRun and shows toast when dry-run toggled", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByTestId("dry-run-toggle"));
    // isDryRun was false → next is true
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Test mode enabled",
      }),
    );
  });
});

describe("ChatInput guided prompt prefill", () => {
  it("prefills the composer and focuses it when an initial prompt arrives after mount", async () => {
    const { rerender } = render(<ChatInput onSend={mockOnSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    textarea.blur();
    expect(document.activeElement).not.toBe(textarea);

    mockInitialPrompt = "Teach me a new skill";
    rerender(<ChatInput onSend={mockOnSend} />);

    await waitFor(() => {
      expect(textarea.value).toBe("Teach me a new skill");
    });
    expect(document.activeElement).toBe(textarea);
    expect(mockSetInitialPrompt).toHaveBeenCalledWith(null);
  });

  it("replaces the current draft when a new guided prompt arrives", async () => {
    const { rerender } = render(<ChatInput onSend={mockOnSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    mockInitialPrompt = NEW_SKILL_PROMPT;
    rerender(<ChatInput onSend={mockOnSend} />);
    await waitFor(() => {
      expect(textarea.value).toBe(NEW_SKILL_PROMPT);
    });

    mockInitialPrompt = NEW_SCHEDULED_TASK_PROMPT;
    rerender(<ChatInput onSend={mockOnSend} />);
    await waitFor(() => {
      expect(textarea.value).toBe(NEW_SCHEDULED_TASK_PROMPT);
    });
  });

  it("clears an untouched guided prompt when the menu discards it", async () => {
    const { rerender } = render(<ChatInput onSend={mockOnSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    mockInitialPrompt = NEW_SKILL_PROMPT;
    rerender(<ChatInput onSend={mockOnSend} />);
    await waitFor(() => {
      expect(textarea.value).toBe(NEW_SKILL_PROMPT);
    });

    fireEvent.click(screen.getByTestId("attachment-menu"));

    await waitFor(() => {
      expect(textarea.value).toBe("");
    });
  });

  it("keeps a user-edited draft when the menu asks to discard", async () => {
    const { rerender } = render(<ChatInput onSend={mockOnSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    mockInitialPrompt = NEW_SKILL_PROMPT;
    rerender(<ChatInput onSend={mockOnSend} />);
    await waitFor(() => {
      expect(textarea.value).toBe(NEW_SKILL_PROMPT);
    });

    fireEvent.change(textarea, {
      target: { value: `${NEW_SKILL_PROMPT} plus my edits` },
    });

    fireEvent.click(screen.getByTestId("attachment-menu"));

    expect(textarea.value).toBe(`${NEW_SKILL_PROMPT} plus my edits`);
  });
});

describe("ChatInput submit behavior", () => {
  it("does not call onSend when textarea is empty", () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const form = screen.getByTestId("submit").closest("form")!;
    fireEvent.submit(form);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("sends trimmed value and clears textarea", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "  hello  " } });
    const form = textarea.closest("form")!;
    fireEvent.submit(form);
    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith("hello", undefined, undefined);
    });
    await waitFor(() => {
      expect(textarea.value).toBe("");
    });
  });

  it("clears the textarea on submit, without waiting for the stream to end", async () => {
    // onSend resolves only when the whole assistant turn finishes, so a
    // clear-after-await left the sent message sitting in the composer for
    // the entire stream.
    let finishStream: (() => void) | undefined;
    const onSend = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishStream = resolve;
        }),
    );
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(textarea.value).toBe("");
    });
    expect(onSend).toHaveBeenCalledWith("hello", undefined, undefined);
    await act(async () => {
      finishStream?.();
    });
  });

  it("clears attachment chips on submit, without waiting for the stream to end", async () => {
    let finishStream: (() => void) | undefined;
    const onSend = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishStream = resolve;
        }),
    );
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.paste(textarea, {
      clipboardData: {
        files: [new File(["png"], "shot.png", { type: "image/png" })],
      },
    });
    // An attachment alone makes the message sendable, so the submit button
    // going back to disabled is proof the chips were dropped.
    await waitFor(() => {
      expect((screen.getByTestId("submit") as HTMLButtonElement).disabled).toBe(
        false,
      );
    });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect((screen.getByTestId("submit") as HTMLButtonElement).disabled).toBe(
        true,
      );
    });
    expect(onSend).toHaveBeenCalledWith("", [expect.any(File)], undefined);
    await act(async () => {
      finishStream?.();
    });
  });

  it("does not call onSend when disabled", () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} disabled />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    const form = textarea.closest("form")!;
    fireEvent.submit(form);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("prevents double-submit while a send is in flight", async () => {
    let resolveFirst: (() => void) | undefined;
    const onSend = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveFirst = resolve;
        }),
    );
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    const form = textarea.closest("form")!;
    fireEvent.submit(form);
    fireEvent.submit(form);
    expect(onSend).toHaveBeenCalledTimes(1);
    await act(async () => {
      resolveFirst?.();
    });
  });

  it("allows sending again after a failed send", async () => {
    let failNext = true;
    const onSend = vi.fn(async () => {
      if (failNext) {
        failNext = false;
        throw new Error("fail");
      }
    });
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    const form = textarea.closest("form")!;
    fireEvent.submit(form);
    await waitFor(() => {
      expect(toast).toHaveBeenCalled();
    });
    fireEvent.change(textarea, { target: { value: "retry" } });
    fireEvent.submit(form);
    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(2);
    });
    expect(onSend).toHaveBeenLastCalledWith("retry", undefined, undefined);
  });
});

describe("ChatInput send failure", () => {
  it("toasts and puts the failed message back in the composer", async () => {
    const onSend = vi.fn().mockRejectedValue(new Error("Backend exploded"));
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(textarea.value).toBe("hello");
    });
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't send message",
        description: expect.stringContaining("Backend exploded"),
        variant: "destructive",
      }),
    );
  });

  it("keeps a draft typed during the stream alongside the failed message", async () => {
    let rejectSend: ((error: Error) => void) | undefined;
    const onSend = vi.fn(
      () =>
        new Promise<void>((_resolve, reject) => {
          rejectSend = reject;
        }),
    );
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "first message" } });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(textarea.value).toBe("");
    });
    fireEvent.change(textarea, { target: { value: "second thought" } });
    await act(async () => {
      rejectSend?.(new Error("nope"));
    });

    expect(textarea.value).toContain("first message");
    expect(textarea.value).toContain("second thought");
  });

  it("does not toast when the send succeeds", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    expect(textarea.value).toBe("");
    expect(toast).not.toHaveBeenCalled();
  });
});

describe("ChatInput clipboard paste", () => {
  function pasteFiles(target: HTMLElement, files: File[]) {
    return fireEvent.paste(target, { clipboardData: { files } });
  }

  it("attaches a pasted image and sends it with the message", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    const image = new File(["png-bytes"], "image.png", { type: "image/png" });
    pasteFiles(textarea, [image]);

    fireEvent.change(textarea, { target: { value: "see screenshot" } });
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    const [message, files, workspaceFiles] = onSend.mock.calls[0];
    expect(message).toBe("see screenshot");
    expect(files).toHaveLength(1);
    expect(files[0].name).toMatch(/^pasted-image-.+\.png$/);
    expect(files[0].type).toBe("image/png");
    expect(workspaceFiles).toBeUndefined();
  });

  it("allows sending a pasted image without any text", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    const image = new File(["png-bytes"], "image.png", { type: "image/png" });
    pasteFiles(textarea, [image]);
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    const [message, files] = onSend.mock.calls[0];
    expect(message).toBe("");
    expect(files).toHaveLength(1);
  });

  it("keeps the original name of pasted non-generic files", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    const pdf = new File(["pdf-bytes"], "report.pdf", {
      type: "application/pdf",
    });
    pasteFiles(textarea, [pdf]);
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    expect(onSend.mock.calls[0][1][0].name).toBe("report.pdf");
  });

  it("does not rename non-image files with generic image names", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    pasteFiles(textarea, [
      new File(["pdf-bytes"], "image.pdf", { type: "application/pdf" }),
    ]);
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    expect(onSend.mock.calls[0][1][0].name).toBe("image.pdf");
  });

  it("gives images from separate pastes in the same second distinct names", async () => {
    vi.useFakeTimers();
    try {
      const baseTime = new Date("2026-01-01T10:00:00.100Z");
      vi.setSystemTime(baseTime);
      const onSend = vi.fn().mockResolvedValue(undefined);
      render(<ChatInput onSend={onSend} />);
      const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

      pasteFiles(textarea, [
        new File(["a"], "image.png", { type: "image/png" }),
      ]);
      vi.setSystemTime(new Date("2026-01-01T10:00:00.900Z"));
      pasteFiles(textarea, [
        new File(["b"], "image.png", { type: "image/png" }),
      ]);

      vi.useRealTimers();
      fireEvent.submit(textarea.closest("form")!);
      await waitFor(() => {
        expect(onSend).toHaveBeenCalledTimes(1);
      });
      const files = onSend.mock.calls[0][1] as File[];
      expect(files).toHaveLength(2);
      expect(files[0].name).not.toBe(files[1].name);
    } finally {
      vi.useRealTimers();
    }
  });

  it("gives multiple generic pasted images distinct names", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea") as HTMLTextAreaElement;

    pasteFiles(textarea, [
      new File(["a"], "image.png", { type: "image/png" }),
      new File(["b"], "image.png", { type: "image/png" }),
    ]);
    fireEvent.submit(textarea.closest("form")!);

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(1);
    });
    const files = onSend.mock.calls[0][1] as File[];
    expect(files).toHaveLength(2);
    expect(files[0].name).not.toBe(files[1].name);
  });

  it("prevents the default paste when files are attached", () => {
    render(<ChatInput onSend={mockOnSend} />);
    const textarea = screen.getByTestId("textarea");
    const image = new File(["png-bytes"], "image.png", { type: "image/png" });
    const notCancelled = pasteFiles(textarea, [image]);
    expect(notCancelled).toBe(false);
  });

  it("leaves plain-text paste untouched", () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByTestId("textarea");
    const notCancelled = pasteFiles(textarea, []);
    expect(notCancelled).toBe(true);
    fireEvent.submit(textarea.closest("form")!);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("does not attach pasted files while uploading", () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatInput onSend={onSend} isUploadingFiles />);
    const textarea = screen.getByTestId("textarea");
    const image = new File(["png-bytes"], "image.png", { type: "image/png" });
    const notCancelled = pasteFiles(textarea, [image]);
    expect(notCancelled).toBe(true);
  });
});

interface StopHarnessProps {
  sessionId: string | null;
  sdkStop: () => void;
  setMessages: (
    updater: ((prev: UIMessage[]) => UIMessage[]) | UIMessage[],
  ) => void;
  setIsUserStopping: (value: boolean) => void;
}

function StopHarness({
  sessionId,
  sdkStop,
  setMessages,
  setIsUserStopping,
}: StopHarnessProps) {
  const isUserStoppingRef = useRef(false);
  const stop = useCopilotStop({
    sessionId,
    sdkStop,
    setMessages: setMessages as Parameters<
      typeof useCopilotStop
    >[0]["setMessages"],
    isUserStoppingRef,
    setIsUserStopping,
  });
  return <ChatInput onSend={vi.fn()} isStreaming onStop={stop} />;
}

function asstMessage(parts: UIMessage["parts"], id = "a1"): UIMessage {
  return { id, role: "assistant", parts };
}

describe("ChatInput stop button", () => {
  it("appends a cancellation marker to the trailing assistant message", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    let messages: UIMessage[] = [
      asstMessage([{ type: "text", text: "partial reply", state: "done" }]),
    ];
    const setMessages = vi.fn((updater: unknown) => {
      if (typeof updater === "function") {
        messages = (updater as (prev: UIMessage[]) => UIMessage[])(messages);
      } else {
        messages = updater as UIMessage[];
      }
    });

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={vi.fn()}
        setMessages={setMessages}
        setIsUserStopping={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      const last = messages[messages.length - 1];
      const tail = last.parts[last.parts.length - 1];
      expect(tail.type).toBe("text");
      expect((tail as { text: string }).text).toContain("Operation cancelled");
    });
  });

  it("leaves messages unchanged when there is no trailing assistant message", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    let messages: UIMessage[] = [
      {
        id: "u1",
        role: "user",
        parts: [{ type: "text", text: "hi", state: "done" }],
      },
    ];
    const setMessages = vi.fn((updater: unknown) => {
      if (typeof updater === "function") {
        messages = (updater as (prev: UIMessage[]) => UIMessage[])(messages);
      } else {
        messages = updater as UIMessage[];
      }
    });

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={vi.fn()}
        setMessages={setMessages}
        setIsUserStopping={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      expect(mockCancel).toHaveBeenCalled();
    });
    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe("user");
  });

  it("aborts the SDK stream and flips the user-stopping flag when clicked", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const sdkStop = vi.fn();
    const setIsUserStopping = vi.fn();

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={sdkStop}
        setMessages={vi.fn()}
        setIsUserStopping={setIsUserStopping}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      expect(sdkStop).toHaveBeenCalledTimes(1);
      expect(setIsUserStopping).toHaveBeenCalledWith(true);
    });
  });

  it("still proceeds with cancel if sdkStop throws", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const sdkStop = vi.fn(() => {
      throw new Error("no fetch in flight");
    });

    const swallowWindow = (e: PromiseRejectionEvent) => e.preventDefault();
    const swallowProcess = () => undefined;
    window.addEventListener("unhandledrejection", swallowWindow);
    process.on("unhandledRejection", swallowProcess);
    try {
      render(
        <StopHarness
          sessionId="sess-1"
          sdkStop={sdkStop}
          setMessages={vi.fn()}
          setIsUserStopping={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByTestId("stop"));

      await waitFor(() => {
        expect(mockCancel).toHaveBeenCalledTimes(1);
      });
    } finally {
      window.removeEventListener("unhandledrejection", swallowWindow);
      process.off("unhandledRejection", swallowProcess);
    }
  });

  it("does not call the cancel endpoint when there is no active sessionId", async () => {
    const setIsUserStopping = vi.fn();
    render(
      <StopHarness
        sessionId={null}
        sdkStop={vi.fn()}
        setMessages={vi.fn()}
        setIsUserStopping={setIsUserStopping}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));
    await waitFor(() => {
      expect(setIsUserStopping).toHaveBeenCalledWith(true);
    });
    expect(mockCancel).not.toHaveBeenCalled();
  });

  it("toasts a 'Stop may take a moment' notice on cancel_published_not_confirmed", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockCancel.mockResolvedValue({
      status: 200,
      data: { reason: "cancel_published_not_confirmed" },
    });

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={vi.fn()}
        setMessages={vi.fn()}
        setIsUserStopping={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Stop may take a moment" }),
      );
    });
  });

  it("toasts a destructive notice when the cancel request rejects", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockCancel.mockRejectedValue(new Error("network down"));

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={vi.fn()}
        setMessages={vi.fn()}
        setIsUserStopping={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not stop the task",
          variant: "destructive",
        }),
      );
    });
  });

  it("does not toast when the cancel succeeds with a normal reason", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });

    render(
      <StopHarness
        sessionId="sess-1"
        sdkStop={vi.fn()}
        setMessages={vi.fn()}
        setIsUserStopping={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByTestId("stop"));

    await waitFor(() => {
      expect(mockCancel).toHaveBeenCalled();
    });
    expect(toast).not.toHaveBeenCalled();
  });
});
