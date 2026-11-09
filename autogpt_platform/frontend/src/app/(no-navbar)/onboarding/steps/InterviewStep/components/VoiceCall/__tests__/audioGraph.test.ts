import {
  type FakeAudioContext,
  fakeMediaStream,
  installVoiceFakes,
  type VoiceFakes,
} from "@/tests/integrations/voice-fakes";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  connectCallAudio,
  createCallContext,
  openMicrophone,
} from "../audioGraph";

let fakes: VoiceFakes;

beforeEach(() => {
  fakes = installVoiceFakes();
});

afterEach(() => {
  vi.useRealTimers();
  fakes.uninstall();
});

async function connected() {
  const context = createCallContext();
  const microphone = fakeMediaStream();
  const audio = await connectCallAudio(context, microphone);
  return {
    audio,
    microphone,
    context: context as unknown as FakeAudioContext,
    mic: fakes.micNode()!,
    player: fakes.playerNode()!,
  };
}

describe("the call's audio", () => {
  it("asks for one echo-cancelled voice channel", async () => {
    await openMicrophone();

    expect(fakes.getUserMedia).toHaveBeenCalledWith({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  });

  it("starts the context in the click that opens it", () => {
    const context = createCallContext() as unknown as FakeAudioContext;

    expect(context.resume).toHaveBeenCalledTimes(1);
  });

  it("captures 40 ms frames at 16 kHz and plays 24 kHz audio, from the worklets the page serves", async () => {
    const { context, mic, player } = await connected();

    expect(context.audioWorklet.addModule.mock.calls).toEqual([
      ["/voice/mic-worklet.js"],
      ["/voice/player-worklet.js"],
    ]);
    expect(mic.options).toEqual({
      processorOptions: { targetRate: 16_000, frameMs: 40 },
    });
    expect(player.options).toEqual({
      numberOfInputs: 0,
      outputChannelCount: [1],
      processorOptions: { sourceRate: 24_000 },
    });
  });

  it("hands each captured frame to its listener", async () => {
    const { audio } = await connected();
    const frames: ArrayBuffer[] = [];
    audio.onFrame((frame) => frames.push(frame));

    fakes.speak();

    expect(frames).toHaveLength(1);
  });

  it("queues Otto's audio and drops it on a flush", async () => {
    const { audio, player } = await connected();
    const pcm = new ArrayBuffer(960);

    audio.play(pcm);
    audio.flush();

    expect(player.port.postMessage.mock.calls).toEqual([
      [pcm, [pcm]],
      [{ type: "flush" }],
    ]);
  });

  it("stops the microphone at once and closes the context once Otto has finished", async () => {
    vi.useFakeTimers();
    const { audio, microphone, context, player } = await connected();
    const frames: ArrayBuffer[] = [];
    audio.onFrame((frame) => frames.push(frame));
    player.port.onmessage?.({
      data: { playedS: 1, bufferedS: 0.8 },
    } as MessageEvent);

    audio.finish();
    fakes.speak();

    expect(microphone.getTracks()[0].stop).toHaveBeenCalled();
    expect(frames).toEqual([]);
    await vi.advanceTimersByTimeAsync(600);
    expect(context.close).not.toHaveBeenCalled();

    player.port.onmessage?.({
      data: { playedS: 1.8, bufferedS: 0 },
    } as MessageEvent);
    await vi.advanceTimersByTimeAsync(300);

    expect(context.close).toHaveBeenCalledTimes(1);
  });

  it("closes everything at once, and plays nothing after", async () => {
    const { audio, context, player } = await connected();

    audio.close();
    audio.close();
    audio.play(new ArrayBuffer(960));
    audio.flush();

    expect(context.close).toHaveBeenCalledTimes(1);
    expect(player.port.postMessage).not.toHaveBeenCalled();
  });
});
