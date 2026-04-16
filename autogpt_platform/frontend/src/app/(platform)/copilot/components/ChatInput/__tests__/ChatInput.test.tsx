import {
  render,
  screen,
  fireEvent,
  cleanup,
  act,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../ChatInput";

let mockCopilotMode = "extended_thinking";
const mockSetCopilotChatMode = vi.fn((mode: string) => {
  mockCopilotMode = mode;
});

let mockCopilotLlmModel = "standard";
const mockSetCopilotLlmModel = vi.fn((model: string) => {
  mockCopilotLlmModel = model;
});

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    copilotMode: mockCopilotMode,
    setCopilotMode: mockSetCopilotChatMode,
    copilotChatMode: mockCopilotMode,
    setCopilotChatMode: mockSetCopilotChatMode,
    copilotLlmModel: mockCopilotLlmModel,
    setCopilotLlmModel: mockSetCopilotLlmModel,
    isDryRun: false,
    setIsDryRun: vi.fn(),
    initialPrompt: null,
    setInitialPrompt: vi.fn(),
  }),
}));

let mockFlagValue = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CHAT_MODE_OPTION: "CHAT_MODE_OPTION" },
  useGetFlag: () => mockFlagValue,
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
  PromptInputBody: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PromptInputFooter: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PromptInputSubmit: ({ disabled }: { disabled?: boolean }) => (
    <button disabled={disabled} data-testid="submit">
      Send
    </button>
  ),
  PromptInputTextarea: (props: {
    id?: string;
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
    disabled?: boolean;
    placeholder?: string;
  }) => (
    <textarea
      id={props.id}
      value={props.value}
      onChange={props.onChange}
      disabled={props.disabled}
      placeholder={props.placeholder}
      data-testid="textarea"
    />
  ),
  PromptInputTools: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="tools">{children}</div>
  ),
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

vi.mock("@/components/ui/input-group", () => ({
  InputGroup: ({
    children,
    className,
  }: {
    children: React.ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
}));

vi.mock("../components/AttachmentMenu", () => ({
  AttachmentMenu: () => <div data-testid="attachment-menu" />,
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
  cleanup();
  vi.clearAllMocks();
  mockCopilotMode = "extended_thinking";
  mockCopilotLlmModel = "standard";
});

describe("ChatInput mode toggle", () => {
  it("does not render mode toggle when flag is disabled", () => {
    mockFlagValue = false;
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.queryByLabelText(/switch to/i)).toBeNull();
  });

  it("renders mode toggle when flag is enabled", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.getByLabelText(/switch to fast mode/i)).toBeDefined();
  });

  it("shows Thinking label in extended_thinking mode", () => {
    mockFlagValue = true;
    mockCopilotMode = "extended_thinking";
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.getByText("Thinking")).toBeDefined();
  });

  it("shows Fast label in fast mode", () => {
    mockFlagValue = true;
    mockCopilotMode = "fast";
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.getByText("Fast")).toBeDefined();
  });

  it("toggles from extended_thinking to fast on click", () => {
    mockFlagValue = true;
    mockCopilotMode = "extended_thinking";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to fast mode/i));
    expect(mockSetCopilotChatMode).toHaveBeenCalledWith("fast");
  });

  it("toggles from fast to extended_thinking on click", () => {
    mockFlagValue = true;
    mockCopilotMode = "fast";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to extended thinking/i));
    expect(mockSetCopilotChatMode).toHaveBeenCalledWith("extended_thinking");
  });

  it("hides toggle buttons when streaming", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} isStreaming />);
    expect(
      screen.queryByLabelText(/switch to (fast|extended thinking) mode/i),
    ).toBeNull();
    expect(
      screen.queryByLabelText(/switch to (advanced|balanced|standard) model/i),
    ).toBeNull();
  });

  it("shows mode toggle when hasSession is true and not streaming", () => {
    // Mode is per-message — can be changed between turns even in an existing session.
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} hasSession />);
    expect(
      screen.queryByLabelText(/switch to (fast|extended thinking) mode/i),
    ).not.toBeNull();
  });

  it("exposes aria-pressed=true in extended_thinking mode", () => {
    mockFlagValue = true;
    mockCopilotMode = "extended_thinking";
    render(<ChatInput onSend={mockOnSend} />);
    const button = screen.getByLabelText(/switch to fast mode/i);
    expect(button.getAttribute("aria-pressed")).toBe("true");
  });

  it("sets aria-pressed=false in fast mode", () => {
    mockFlagValue = true;
    mockCopilotMode = "fast";
    render(<ChatInput onSend={mockOnSend} />);
    const button = screen.getByLabelText(/switch to extended thinking/i);
    expect(button.getAttribute("aria-pressed")).toBe("false");
  });

  it("shows a toast when the user toggles mode", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockFlagValue = true;
    mockCopilotMode = "extended_thinking";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to fast mode/i));
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: expect.stringMatching(/switched to fast mode/i),
      }),
    );
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

describe("ChatInput model toggle", () => {
  it("renders model toggle button when flag is enabled", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.getByLabelText(/switch to advanced model/i)).toBeDefined();
  });

  it("does not render model toggle when flag is disabled", () => {
    mockFlagValue = false;
    render(<ChatInput onSend={mockOnSend} />);
    expect(
      screen.queryByLabelText(/switch to (advanced|standard) model/i),
    ).toBeNull();
  });

  it("toggles from standard to advanced on click", () => {
    mockFlagValue = true;
    mockCopilotLlmModel = "standard";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to advanced model/i));
    expect(mockSetCopilotLlmModel).toHaveBeenCalledWith("advanced");
  });

  it("toggles from advanced to standard on click", () => {
    mockFlagValue = true;
    mockCopilotLlmModel = "advanced";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to balanced model/i));
    expect(mockSetCopilotLlmModel).toHaveBeenCalledWith("standard");
  });

  it("hides model toggle when streaming", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} isStreaming />);
    expect(
      screen.queryByLabelText(/switch to (advanced|standard) model/i),
    ).toBeNull();
  });

  it("shows model toggle when hasSession is true and not streaming", () => {
    // Model is per-message — can be changed between turns even in an existing session.
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} hasSession />);
    expect(
      screen.queryByLabelText(/switch to (advanced|standard) model/i),
    ).not.toBeNull();
  });

  it("hides dry-run toggle when hasSession is true", () => {
    // DryRun button is only for new chats — once a session exists its dry_run
    // flag is immutable and shown via the CopilotPage banner, not this button.
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} hasSession />);
    expect(screen.queryByTestId("dry-run-toggle")).toBeNull();
  });

  it("shows dry-run toggle when no session", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} />);
    expect(screen.getByTestId("dry-run-toggle")).toBeTruthy();
  });

  it("shows a toast when switching to advanced", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockFlagValue = true;
    mockCopilotLlmModel = "standard";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to advanced model/i));
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: expect.stringMatching(/switched to advanced model/i),
      }),
    );
  });

  it("shows a toast when switching to standard", async () => {
    const { toast } = await import("@/components/molecules/Toast/use-toast");
    mockFlagValue = true;
    mockCopilotLlmModel = "advanced";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to balanced model/i));
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: expect.stringMatching(/switched to balanced model/i),
      }),
    );
  });
});
