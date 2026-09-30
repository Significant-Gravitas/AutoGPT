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
// the content box at CHAR_WIDTH per character.
//
// The box is `box-sizing: border-box` with horizontal padding, so the two ways
// of reading its width disagree and an inline width means the border box:
// getBoundingClientRect and `style.width` speak border box, getComputedStyle
// speaks content box. Modelling that gap is what makes replaying a content-box
// width as an inline one observable here rather than only in a browser.
const ADDONS_WIDTH = 240;
const CHAR_WIDTH = 10;
const LINE_HEIGHT = 24;
const PADDING = 12;
const PADDING_X = 12;
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

// An inline `style.width` sets the border box, so it wins over the layout.
function borderBoxWidth(textarea: HTMLTextAreaElement) {
  const inlineWidth = parseFloat(textarea.style.width);
  return Number.isFinite(inlineWidth) ? inlineWidth : layoutWidth(textarea);
}

function contentBoxWidth(borderBox: number) {
  return borderBox - PADDING_X * 2;
}

function contentWidth(textarea: HTMLTextAreaElement) {
  return contentBoxWidth(borderBoxWidth(textarea));
}

function contentHeight(textarea: HTMLTextAreaElement) {
  const content = textarea.value || textarea.placeholder;
  const limit = charsPerLine(contentWidth(textarea));
  const lines = Math.max(1, Math.ceil(content.length / limit));
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
    width: `${contentWidth(textarea)}px`,
  } as unknown as CSSStyleDeclaration;
}

// The browser reports the new width after the host re-lays out the row;
// here the test delivers that notification. `contentRect` is the content box,
// which is all the component uses it for: noticing that the width changed.
function reportResize(textarea: HTMLTextAreaElement) {
  act(() => {
    const entry = {
      target: textarea,
      contentRect: { width: contentBoxWidth(layoutWidth(textarea)) },
    } as unknown as ResizeObserverEntry;
    resizeCallbacks.forEach((callback) =>
      callback([entry], {} as ResizeObserver),
    );
  });
}

// Every layout the host puts the composer through, not just the one it lands
// on. A wrong measurement that corrects itself on the next frame still settles
// on the right layout, so the settled state alone cannot tell a fix from the
// blink it was meant to remove; the record of what the row passed through can.
function recordLayouts(textarea: HTMLTextAreaElement) {
  const row = textarea.parentElement as HTMLElement;
  const passedThrough: boolean[] = [];
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      passedThrough.push((record.oldValue ?? "").split(" ").includes("w-full"));
    }
  });
  observer.observe(row, {
    attributes: true,
    attributeFilter: ["class"],
    attributeOldValue: true,
  });
  return {
    // The states it left, plus the one it is in now.
    seen: () => [...passedThrough, isStacked(textarea)],
    stop: () => observer.disconnect(),
  };
}

function renderComposer(placeholder: string) {
  render(<ChatInput onSend={vi.fn()} placeholder={placeholder} />);
  return screen.getByLabelText("Chat message input") as HTMLTextAreaElement;
}

// How many characters fit one line of a row of the given border-box width.
function fitsInRow(borderBox: number) {
  return charsPerLine(contentBoxWidth(borderBox));
}

// Text that wraps in the single row but fits the full-width one: the band the
// composer used to shake in.
function wrapsInRowOnly() {
  return "x".repeat(fitsInRow(singleRowWidth()) + 10);
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
  Object.defineProperty(
    HTMLTextAreaElement.prototype,
    "getBoundingClientRect",
    {
      configurable: true,
      value: function (this: HTMLTextAreaElement) {
        return { width: borderBoxWidth(this) } as unknown as DOMRect;
      },
    },
  );
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
  Reflect.deleteProperty(
    HTMLTextAreaElement.prototype,
    "getBoundingClientRect",
  );
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
      target: { value: "x".repeat(fitsInRow(singleRowWidth()) - 10) },
    });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(false);
  });

  it("keeps the empty hero composer stacked when only its placeholder wraps in the row", () => {
    const textarea = renderComposer(
      "p".repeat(fitsInRow(singleRowWidth()) + 10),
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
    expect(fitsInRow(singleRowWidth())).toBeGreaterThan(value.length);
  });

  it("never measures against a single row wider than the row it now has", async () => {
    const textarea = renderComposer("Ask");
    fireEvent.change(textarea, { target: { value: wrapsInRowOnly() } });
    expect(isStacked(textarea)).toBe(true);

    // The window narrows past what the addons themselves take, so subtracting
    // them from the row leaves nothing: the offset can say nothing about this
    // row, and the remembered width now describes one wider than the whole
    // composer.
    const remembered = singleRowWidth();
    rowWidth = ADDONS_WIDTH - 40;
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);

    // Text that would fit the row the box remembers but wraps in the one it
    // actually has. Judged against the stale width the composer unstacks, then
    // wraps in the real row on the very next measurement and stacks again.
    const value = "x".repeat(fitsInRow(rowWidth) + 4);
    expect(value.length).toBeLessThan(fitsInRow(remembered));

    const layouts = recordLayouts(textarea);
    fireEvent.change(textarea, { target: { value } });
    await act(async () => undefined);
    layouts.stop();

    expect(layouts.seen()).toEqual([true]);
  });

  it("measures the single row at its border-box width", () => {
    const textarea = renderComposer("Ask");
    fireEvent.change(textarea, { target: { value: wrapsInRowOnly() } });
    reportResize(textarea);
    expect(isStacked(textarea)).toBe(true);

    // Text that fits the real single row but not one mistaken for its content
    // box: a content-box width handed back as an inline width loses the box's
    // horizontal padding, so the row is judged PADDING_X * 2 too narrow.
    const fits = fitsInRow(singleRowWidth());
    expect(fits).toBeGreaterThan(fitsInRow(contentBoxWidth(singleRowWidth())));

    fireEvent.change(textarea, { target: { value: "x".repeat(fits) } });
    expect(isStacked(textarea)).toBe(false);
  });
});
