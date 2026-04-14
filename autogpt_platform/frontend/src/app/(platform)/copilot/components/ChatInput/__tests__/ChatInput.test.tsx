import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../ChatInput";

let mockCopilotMode = "extended_thinking";
const mockSetCopilotMode = vi.fn((mode: string) => {
  mockCopilotMode = mode;
});

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    copilotMode: mockCopilotMode,
    setCopilotMode: mockSetCopilotMode,
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

const mockOnSend = vi.fn();

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  mockCopilotMode = "extended_thinking";
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
    expect(mockSetCopilotMode).toHaveBeenCalledWith("fast");
  });

  it("toggles from fast to extended_thinking on click", () => {
    mockFlagValue = true;
    mockCopilotMode = "fast";
    render(<ChatInput onSend={mockOnSend} />);
    fireEvent.click(screen.getByLabelText(/switch to extended thinking/i));
    expect(mockSetCopilotMode).toHaveBeenCalledWith("extended_thinking");
  });

  it("hides toggle button when streaming", () => {
    mockFlagValue = true;
    render(<ChatInput onSend={mockOnSend} isStreaming />);
    expect(screen.queryByLabelText(/switch to/i)).toBeNull();
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
