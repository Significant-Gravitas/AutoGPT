import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE_RATE = 24_000;

interface Player {
  size: number;
  read: number;
  written: number;
  port: { onmessage: (event: { data: unknown }) => void };
  process: (inputs: unknown[], outputs: Float32Array[][]) => boolean;
}

type PlayerClass = new (options: unknown) => Player;

class FakeProcessor {
  port = { onmessage: null, postMessage: () => undefined };
}

// The player runs in the audio thread's own scope; its globals are handed
// in here instead.
function newPlayer(): Player {
  const source = readFileSync(
    resolve(process.cwd(), "public/voice/player-worklet.js"),
    "utf8",
  );
  let Processor: PlayerClass | null = null;
  const load = new Function(
    "AudioWorkletProcessor",
    "registerProcessor",
    "sampleRate",
    source,
  );
  load(
    FakeProcessor,
    (_name: string, processor: PlayerClass) => {
      Processor = processor;
    },
    SOURCE_RATE,
  );
  if (!Processor) throw new Error("The player registered no processor");
  return new (Processor as PlayerClass)({
    processorOptions: { sourceRate: SOURCE_RATE },
  });
}

// Seconds of audio, each sample its own place in the call, wrapped to fit
// PCM16.
function audio(seconds: number, from = 0) {
  const pcm = new Int16Array(seconds * SOURCE_RATE);
  for (let i = 0; i < pcm.length; i++) pcm[i] = (from + i) % 16_384;
  return pcm.buffer;
}

function play(player: Player) {
  const out = new Float32Array(128);
  player.process([], [[out]]);
  return out;
}

describe("the call's player", () => {
  it("holds 30 s of Otto's audio: a re-attach's replay of up to 20 s, and more on its way", () => {
    const player = newPlayer();

    player.port.onmessage({ data: audio(25) });

    expect(player.size).toBe(30 * SOURCE_RATE);
    expect(player.written - player.read).toBe(25 * SOURCE_RATE);
  });

  it("drops the oldest audio waiting, never reading over it, once more arrives than it holds", () => {
    const player = newPlayer();

    player.port.onmessage({ data: audio(20) });
    player.port.onmessage({ data: audio(15, 20 * SOURCE_RATE) });

    expect(player.written - player.read).toBe(player.size - 1);
    const oldestKept = player.read;
    expect(play(player)[0]).toBeCloseTo((oldestKept % 16_384) / 32_768, 6);
  });

  it("drops everything waiting when Otto is interrupted", () => {
    const player = newPlayer();
    player.port.onmessage({ data: audio(5) });

    player.port.onmessage({ data: { type: "flush" } });

    expect(player.written - player.read).toBe(0);
    expect(play(player).every((sample) => sample === 0)).toBe(true);
  });
});
