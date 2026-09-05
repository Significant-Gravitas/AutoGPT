import {
  cleanup,
  render as rtlRender,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Hoisted: `vi.mock` factories are lifted above every top-level binding.
const { synthesizeSpeech, useGetFlag } = vi.hoisted(() => ({
  synthesizeSpeech: vi.fn(
    async (text: string, sessionID: string | null, kind?: string) =>
      new Blob([`${kind}:${sessionID}:${text}`]),
  ),
  useGetFlag: vi.fn(() => true),
}));

vi.mock("../../../../voice/speechApi", () => ({
  synthesizeSpeech,
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { COPILOT_VOICE_MODE: "copilot-voice-mode" },
  useGetFlag: () => useGetFlag(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { TTSButton } from "../TTSButton";

function render(ui: React.ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

describe("TTSButton", () => {
  beforeEach(() => {
    stubSpeechSynthesis([]);
    stubAudio();
    global.URL.createObjectURL = vi.fn(() => "blob:chunk");
    global.URL.revokeObjectURL = vi.fn();
    useGetFlag.mockReturnValue(true);
    synthesizeSpeech.mockClear();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("speaks through the browser when it has voices, without spending on synthesis", async () => {
    const speak = stubSpeechSynthesis([{ name: "Samantha", lang: "en-US" }]);
    render(<TTSButton text="A local voice reads this." sessionID="s-1" />);

    await userEvent.click(await screen.findByRole("button"));

    expect(speak).toHaveBeenCalledTimes(1);
    expect(synthesizeSpeech).not.toHaveBeenCalled();
  });

  it("falls back to our own synthesis when the browser has no voices", async () => {
    render(<TTSButton text="No local voice reads this." sessionID="s-1" />);

    await userEvent.click(await screen.findByRole("button"));

    await waitFor(() => expect(synthesizeSpeech).toHaveBeenCalled());
    expect(synthesizeSpeech.mock.calls[0]).toEqual([
      "No local voice reads this.",
      "s-1",
      "reply",
    ]);
  });

  it("splits a reply too long for one request into several", async () => {
    const long = "This sentence is here to be spoken aloud. ".repeat(30);
    render(<TTSButton text={long} sessionID={null} />);

    await userEvent.click(await screen.findByRole("button"));

    await waitFor(() => expect(synthesizeSpeech).toHaveBeenCalled());
    const chunks = synthesizeSpeech.mock.calls.map((call) => call[0]);
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((chunk) => expect(chunk.length).toBeLessThanOrEqual(4096));
  });

  it("attributes the cost to the session the message belongs to", async () => {
    render(<TTSButton text="Bill this to the session." sessionID="s-42" />);

    await userEvent.click(await screen.findByRole("button"));

    await waitFor(() => expect(synthesizeSpeech).toHaveBeenCalled());
    expect(synthesizeSpeech.mock.calls[0][1]).toBe("s-42");
  });

  it("hides itself when it has neither voices nor the flag that lets it synthesise", () => {
    useGetFlag.mockReturnValue(false);
    render(<TTSButton text="Nothing can speak this." sessionID="s-1" />);

    expect(screen.queryByRole("button")).toBeNull();
  });

  it("still offers browser speech when the flag is off but voices exist", async () => {
    useGetFlag.mockReturnValue(false);
    stubSpeechSynthesis([{ name: "Samantha", lang: "en-US" }]);
    render(<TTSButton text="A local voice reads this." sessionID="s-1" />);

    // findByRole rejects when nothing matches, so this is a real presence check.
    expect(await screen.findByRole("button")).not.toBeNull();
  });
});

function stubSpeechSynthesis(voices: Partial<SpeechSynthesisVoice>[]) {
  const speak = vi.fn();
  vi.stubGlobal(
    "SpeechSynthesisUtterance",
    class {
      constructor(public text: string) {}
    },
  );
  vi.stubGlobal("speechSynthesis", {
    getVoices: () => voices as SpeechSynthesisVoice[],
    speak,
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  return speak;
}

function stubAudio() {
  vi.stubGlobal(
    "Audio",
    class {
      src = "";
      preload = "";
      play = vi.fn(async () => undefined);
      pause = vi.fn();
      addEventListener = vi.fn(
        (event: string, handler: () => void) =>
          event === "ended" && setTimeout(handler, 0),
      );
      removeEventListener = vi.fn();
    },
  );
}
