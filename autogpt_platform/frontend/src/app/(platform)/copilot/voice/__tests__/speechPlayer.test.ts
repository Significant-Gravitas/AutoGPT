import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createSpeechPlayer } from "../speechPlayer";

describe("createSpeechPlayer", () => {
  beforeEach(() => {
    stubAudio();
    global.URL.createObjectURL = vi.fn(() => "blob:chunk");
    global.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => vi.restoreAllMocks());

  it("plays chunks in the order they were enqueued", async () => {
    const played: string[] = [];
    const player = createSpeechPlayer({
      synthesize: async (text) => {
        played.push(`synth:${text}`);
        return new Blob([text]);
      },
      onIdle: () => played.push("idle"),
      onError: () => undefined,
    });

    player.enqueue("one");
    player.enqueue("two");
    await settle();

    expect(played).toEqual(["synth:one", "synth:two", "idle"]);
  });

  it("reports idle only once the queue has drained", async () => {
    const onIdle = vi.fn();
    const player = createSpeechPlayer({
      synthesize: async () => new Blob(["x"]),
      onIdle,
      onError: () => undefined,
    });

    player.enqueue("one");
    expect(player.isIdle()).toBe(false);
    await settle();

    expect(player.isIdle()).toBe(true);
    expect(onIdle).toHaveBeenCalledTimes(1);
  });

  it("drops audio synthesised for a turn that was stopped", async () => {
    let release: (blob: Blob) => void = () => undefined;
    const onIdle = vi.fn();
    const player = createSpeechPlayer({
      synthesize: () => new Promise<Blob>((resolve) => (release = resolve)),
      onIdle,
      onError: () => undefined,
    });

    player.enqueue("abandoned");
    player.stop();
    release(new Blob(["late"]));
    await settle();

    expect(global.URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("keeps playing after a stop when new chunks arrive", async () => {
    const played: string[] = [];
    const player = createSpeechPlayer({
      synthesize: async (text) => new Blob([text]),
      onIdle: () => played.push("idle"),
      onError: () => undefined,
    });

    player.enqueue("first");
    player.stop();
    player.enqueue("second");
    await settle();

    expect(played).toContain("idle");
    expect(player.isIdle()).toBe(true);
  });

  it("keeps working after a stop lands mid-playback", async () => {
    // stop() swaps the element's src, which fires neither "ended" nor
    // "error". Leaving that playback unsettled wedges the queue for the life
    // of the tab: state says Speaking, no audio ever comes.
    vi.restoreAllMocks();
    vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(
      undefined,
    );
    const played: string[] = [];
    const player = createSpeechPlayer({
      synthesize: async (text) => {
        played.push(text);
        return new Blob([text]);
      },
      onIdle: () => undefined,
      onError: () => undefined,
    });

    player.enqueue("first");
    await settle();
    expect(player.isIdle()).toBe(false);

    player.stop();
    await settle();
    expect(player.isIdle()).toBe(true);

    stubAudio();
    player.enqueue("second");
    await settle();
    expect(played).toContain("second");
    expect(player.isIdle()).toBe(true);
  });

  it("reports a failed synthesis and moves on to the next chunk", async () => {
    const onError = vi.fn();
    const player = createSpeechPlayer({
      synthesize: async (text) => {
        if (text === "bad") throw new Error("503");
        return new Blob([text]);
      },
      onIdle: () => undefined,
      onError,
    });

    player.enqueue("bad");
    player.enqueue("good");
    await settle();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(global.URL.createObjectURL).toHaveBeenCalledTimes(1);
  });
});

/** jsdom has no media pipeline: play() resolves and "ended" fires on demand. */
function stubAudio() {
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(
    function (this: HTMLAudioElement) {
      queueMicrotask(() => this.dispatchEvent(new Event("ended")));
      return Promise.resolve();
    },
  );
}

async function settle() {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}
