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

// jsdom lays nothing out, so the composer's geometry is played here: the box
// shares a row with the addons (the row minus ADDONS_WIDTH) until the host
// stacks it on its own full-width row, and text wraps once it no longer fits
// the width being measured at CHAR_WIDTH per character.
const ADDONS_WIDTH = 240;
const CHAR_WIDTH = 10;
const LINE_HEIGHT = 24;
const PADDING = 12;
// The hero composer floors the box at 4.5rem, so scrollHeight reports that
// floor for content that would be shorter — until measureContentHeight lifts
// it for the read.
const HERO_MIN_HEIGHT = 72;

const DEFAULT_ROW_WIDTH = 640;
let rowWidth = DEFAULT_ROW_WIDTH;

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

function singleRowWidth() {
  return rowWidth - ADDONS_WIDTH;
}

function charsPerLine(width: number) {
  return Math.max(1, Math.floor(width / CHAR_WIDTH));
}

function isStacked(textarea: HTMLTextAreaElement) {
  return textarea.parentElement?.classList.contains("w-full") ?? false;
}

function layoutWidth(textarea: HTMLTextAreaElement) {
  return isStacked(textarea) ? rowWidth : singleRowWidth();
}

function measuredWidth(textarea: HTMLTextAreaElement) {
  const inlineWidth = parseFloat(textarea.style.width);
  return Number.isFinite(inlineWidth) ? inlineWidth : layoutWidth(textarea);
}

function contentHeight(textarea: HTMLTextAreaElement) {
  const content = textarea.value || textarea.placeholder;
  const lines = Math.max(
    1,
    Math.ceil(content.length / charsPerLine(measuredWidth(textarea))),
  );
  const height = lines * LINE_HEIGHT + PADDING * 2;
  return parseFloat(textarea.style.minHeight) === 0
    ? height
    : Math.max(height, HERO_MIN_HEIGHT);
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

// Text that wraps in the single row but fits the full-width one: the band the
// composer used to shake in.
function wrapsInRowOnly() {
  return "x".repeat(charsPerLine(singleRowWidth()) + 10);
}

beforeEach(() => {
  rowWidth = DEFAULT_ROW_WIDTH;
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

    fireEvent.change(textarea, { target: { value: wrapsInRowOnly() } });
    expect(isStacked(textarea)).toBe(true);

    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
  });

  it("returns to the single row once the text fits it again", () => {
    const textarea = renderComposer("Ask");
    fireEvent.change(textarea, { target: { value: wrapsInRowOnly() } });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);

    fireEvent.change(textarea, {
      target: { value: "x".repeat(charsPerLine(singleRowWidth()) - 10) },
    });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(false);
  });

  it("keeps the empty hero composer stacked when only its placeholder wraps in the row", () => {
    const textarea = renderComposer(
      "p".repeat(charsPerLine(singleRowWidth()) + 10),
    );
    expect(isStacked(textarea)).toBe(true);

    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);
  });

  it("re-measures the single row when the host row grows while stacked", () => {
    const textarea = renderComposer("Ask");
    const value = wrapsInRowOnly();
    fireEvent.change(textarea, { target: { value } });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);

    // A side panel closes: the same text now fits the wider single row, so the
    // box must judge itself against that row rather than the remembered one.
    rowWidth = DEFAULT_ROW_WIDTH * 2;
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(false);
    expect(charsPerLine(singleRowWidth())).toBeGreaterThan(value.length);
  });
});
