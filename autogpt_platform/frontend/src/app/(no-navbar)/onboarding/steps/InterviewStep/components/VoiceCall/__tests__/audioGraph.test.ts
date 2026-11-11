import {
  type FakeAudioContext,
  fakeMediaStream,
  installVoiceFakes,
  type VoiceFakes,
} from "@/tests/integrations/voice-fakes";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  createCallAudio,
  createCallContext,
  createCallPlayer,
  openMicrophone,
  releasePlayer,
} from "../audioGraph";

let fakes: VoiceFakes;

beforeEach(() => {
  fakes = installVoiceFakes();
});

afterEach(() => {
  vi.useRealTimers();
  fakes.uninstall();
});

const OTTO = { kind: "audio", id: "otto-track" } as unknown as MediaStreamTrack;

function callAudio() {
  const context = createCallContext();
  const microphone = fakeMediaStream();
  const audio = createCallAudio(context, microphone);
  return {
    audio,
    microphone,
    context: context as unknown as FakeAudioContext,
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

  it("starts Otto's player on silence in the click, and lets it go", () => {
    const context = createCallContext();
    const player = createCallPlayer(context);

    expect(player.autoplay).toBe(true);
    expect(player.srcObject).not.toBeNull();

    releasePlayer(player);
    expect(player.srcObject).toBeNull();
  });

  it("mixes the person's voice into the levels the aura moves with", () => {
    const { audio } = callAudio();

    expect(audio.levels).toBeInstanceOf(MediaStream);
  });

  it("says once when Otto's speech first carries sound", async () => {
    vi.useFakeTimers();
    const { audio } = callAudio();
    const onSound = vi.fn();
    audio.listen(OTTO, onSound);

    await vi.advanceTimersByTimeAsync(200);
    expect(onSound).not.toHaveBeenCalled();

    fakes.ottoSounds();
    await vi.advanceTimersByTimeAsync(60);
    await vi.advanceTimersByTimeAsync(500);

    expect(onSound).toHaveBeenCalledTimes(1);
  });

  it("stops listening for Otto's sound once the call's audio is closed", async () => {
    vi.useFakeTimers();
    const { audio } = callAudio();
    const onSound = vi.fn();
    audio.listen(OTTO, onSound);

    audio.close();
    fakes.ottoSounds();
    await vi.advanceTimersByTimeAsync(500);

    expect(onSound).not.toHaveBeenCalled();
  });

  it("releases the microphone at once on mute, and keeps the context", () => {
    const { audio, microphone, context } = callAudio();

    audio.mute();
    audio.mute();

    expect(microphone.getTracks()[0].stop).toHaveBeenCalledTimes(1);
    expect(context.close).not.toHaveBeenCalled();
  });

  it("closes everything once, and mixes nothing in after", () => {
    const { audio, microphone, context } = callAudio();

    audio.close();
    audio.close();
    audio.listen(OTTO, vi.fn());

    expect(microphone.getTracks()[0].stop).toHaveBeenCalledTimes(1);
    expect(context.close).toHaveBeenCalledTimes(1);
  });
});
