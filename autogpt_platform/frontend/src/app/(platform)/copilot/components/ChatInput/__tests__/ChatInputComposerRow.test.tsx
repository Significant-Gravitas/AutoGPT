import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../ChatInput";

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: vi.fn(),
  useGetV2ListChatTransports: () => ({
    data: {
      status: 200,
      data: {
        transports: [
          {
            auth_provider: "platform",
            credential_id: null,
            label: "AutoGPT Platform",
            available: true,
            default: true,
          },
        ],
      },
    },
    isPending: false,
    isError: false,
  }),
}));

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    copilotChatMode: "extended_thinking",
    setCopilotChatMode: vi.fn(),
    copilotModePinned: false,
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
    CHAT_WORKSPACE_FILES: "CHAT_WORKSPACE_FILES",
  },
  useGetFlag: () => false,
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

vi.mock("../components/ComposerPlusMenu", () => ({
  ComposerPlusMenu: () => null,
}));
vi.mock("../components/FileChips", () => ({
  FileChips: () => null,
}));

// jsdom lays nothing out, so the composer's two textarea widths are played
// here: the box shares a row with the addons (narrow) until the host stacks
// it on its own full-width row (wide). Content wraps in the narrow row past
// NARROW_LIMIT characters and in the wide one past WIDE_LIMIT.
const NARROW_WIDTH = 400;
const WIDE_WIDTH = 640;
const NARROW_LIMIT = 40;
const WIDE_LIMIT = 70;
const LINE_HEIGHT = 24;
const PADDING = 12;

const resizeCallbacks: ResizeObserverCallback[] = [];

class ResizeObserverStub {
  constructor(callback: ResizeObserverCallback) {
    resizeCallbacks.push(callback);
  }
  observe() {
    return;
  }
  unobserve() {
    return;
  }
  disconnect() {
    return;
  }
}

function isStacked(textarea: HTMLTextAreaElement) {
  return textarea.parentElement?.classList.contains("w-full") ?? false;
}

function layoutWidth(textarea: HTMLTextAreaElement) {
  return isStacked(textarea) ? WIDE_WIDTH : NARROW_WIDTH;
}

function measuredWidth(textarea: HTMLTextAreaElement) {
  const inlineWidth = parseFloat(textarea.style.width);
  return Number.isFinite(inlineWidth) ? inlineWidth : layoutWidth(textarea);
}

function contentHeight(textarea: HTMLTextAreaElement) {
  const limit =
    measuredWidth(textarea) >= WIDE_WIDTH ? WIDE_LIMIT : NARROW_LIMIT;
  const content = textarea.value || textarea.placeholder;
  const lines = content.length > limit ? 2 : 1;
  return lines * LINE_HEIGHT + PADDING * 2;
}

function fakeTextareaStyle(textarea: HTMLTextAreaElement) {
  return {
    lineHeight: `${LINE_HEIGHT}px`,
    paddingTop: `${PADDING}px`,
    paddingBottom: `${PADDING}px`,
    width: `${measuredWidth(textarea)}px`,
  } as unknown as CSSStyleDeclaration;
}

// The browser reports the new width after the host re-lays out the row;
// here the test delivers that notification.
function reportResize(textarea: HTMLTextAreaElement) {
  act(() => {
    const entry = {
      target: textarea,
      contentRect: { width: layoutWidth(textarea) },
    } as unknown as ResizeObserverEntry;
    resizeCallbacks.forEach((callback) =>
      callback([entry], {} as ResizeObserver),
    );
  });
}

function renderComposer(placeholder: string) {
  render(<ChatInput onSend={vi.fn()} placeholder={placeholder} />);
  return screen.getByLabelText("Chat message input") as HTMLTextAreaElement;
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", ResizeObserverStub);
  Object.defineProperty(HTMLTextAreaElement.prototype, "scrollHeight", {
    configurable: true,
    get(this: HTMLTextAreaElement) {
      return contentHeight(this);
    },
  });
  const realGetComputedStyle = window.getComputedStyle.bind(window);
  vi.spyOn(window, "getComputedStyle").mockImplementation((element, pseudo) =>
    element instanceof HTMLTextAreaElement
      ? fakeTextareaStyle(element)
      : realGetComputedStyle(element, pseudo),
  );
});

afterEach(() => {
  resizeCallbacks.length = 0;
  Reflect.deleteProperty(HTMLTextAreaElement.prototype, "scrollHeight");
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ChatInput composer row layout", () => {
  it("stays stacked once text wraps in the row, even though it fits the full width", () => {
    const textarea = renderComposer("Ask");
    expect(isStacked(textarea)).toBe(false);

    fireEvent.change(textarea, {
      target: { value: "x".repeat(NARROW_LIMIT + 10) },
    });
    expect(isStacked(textarea)).toBe(true);

    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
  });

  it("returns to the single row once the text fits it again", () => {
    const textarea = renderComposer("Ask");
    fireEvent.change(textarea, {
      target: { value: "x".repeat(NARROW_LIMIT + 10) },
    });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);

    fireEvent.change(textarea, {
      target: { value: "x".repeat(NARROW_LIMIT - 10) },
    });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(false);
  });

  it("keeps the empty hero composer stacked when only its placeholder wraps in the row", () => {
    const textarea = renderComposer("p".repeat(NARROW_LIMIT + 10));
    expect(isStacked(textarea)).toBe(true);

    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
  });
});
