import { MIC_FRAME_MS, MIC_RATE, PLAYBACK_RATE } from "./protocol";

const MIC_WORKLET_URL = "/voice/mic-worklet.js";
const PLAYER_WORKLET_URL = "/voice/player-worklet.js";
// Otto's last words play out before the context closes, for at most this long.
const DRAIN_POLL_MS = 250;
const DRAINED_S = 0.05;
const MAX_DRAIN_MS = 25_000;

export interface PlaybackReport {
  playedS: number;
  bufferedS: number;
}

export interface CallAudio {
  // The person's voice and Otto's mixed, for the call's aura.
  levels: MediaStream;
  play(pcm: ArrayBuffer): void;
  // Drops what is queued, when the person talks over Otto.
  flush(): void;
  onFrame(listener: ((frame: ArrayBuffer) => void) | null): void;
  // Mixes Otto's speech from the call's room into the levels; the room's
  // player plays it.
  listen(track: MediaStreamTrack): void;
  // Releases the microphone now, and keeps playing what Otto still says.
  mute(): void;
  // Releases the microphone and closes the context once Otto has finished.
  finish(): void;
  close(): void;
}

// Browsers only let a page start audio from a gesture, so the context is made
// inside the click that starts or rejoins the call, before anything awaits.
export function createCallContext() {
  const context = new AudioContext({ latencyHint: "interactive" });
  void context.resume().catch(() => undefined);
  return context;
}

// On the LiveKit leg Otto's speech plays through an audio element, which a
// browser lets play only once a gesture has started it: it is made, and
// started on silence, inside the same click.
export function createCallPlayer(context: AudioContext) {
  const player = new Audio();
  player.autoplay = true;
  player.srcObject = context.createMediaStreamDestination().stream;
  void player.play().catch(() => undefined);
  return player;
}

export function releasePlayer(player: HTMLAudioElement) {
  player.pause();
  player.srcObject = null;
}

export function openMicrophone() {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
}

export async function connectCallAudio(
  context: AudioContext,
  microphone: MediaStream,
): Promise<CallAudio> {
  await context.audioWorklet.addModule(MIC_WORKLET_URL);
  await context.audioWorklet.addModule(PLAYER_WORKLET_URL);

  const source = context.createMediaStreamSource(microphone);
  const capture = new AudioWorkletNode(context, "mic-capture", {
    processorOptions: { targetRate: MIC_RATE, frameMs: MIC_FRAME_MS },
  });
  // A worklet only runs while its output is pulled, so the capture feeds the
  // speakers through a silent gain.
  const silence = context.createGain();
  silence.gain.value = 0;
  source.connect(capture).connect(silence).connect(context.destination);

  const player = new AudioWorkletNode(context, "pcm-player", {
    numberOfInputs: 0,
    outputChannelCount: [1],
    processorOptions: { sourceRate: PLAYBACK_RATE },
  });
  player.connect(context.destination);

  const mix = context.createMediaStreamDestination();
  source.connect(mix);
  player.connect(mix);

  let frameListener: ((frame: ArrayBuffer) => void) | null = null;
  let bufferedS = 0;
  let isMicrophoneOpen = true;
  let isClosed = false;
  const roomSources: MediaStreamAudioSourceNode[] = [];

  capture.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
    frameListener?.(event.data);
  };
  player.port.onmessage = (event: MessageEvent<PlaybackReport>) => {
    bufferedS = event.data.bufferedS;
  };

  function stopMicrophone() {
    if (!isMicrophoneOpen) return;
    isMicrophoneOpen = false;
    frameListener = null;
    microphone.getTracks().forEach((track) => track.stop());
    source.disconnect();
    capture.disconnect();
  }

  function close() {
    if (isClosed) return;
    isClosed = true;
    stopMicrophone();
    player.disconnect();
    roomSources.forEach((source) => source.disconnect());
    void context.close().catch(() => undefined);
  }

  return {
    levels: mix.stream,
    play(pcm) {
      if (!isClosed) player.port.postMessage(pcm, [pcm]);
    },
    flush() {
      if (!isClosed) player.port.postMessage({ type: "flush" });
    },
    onFrame(listener) {
      frameListener = isMicrophoneOpen ? listener : null;
    },
    listen(track) {
      if (isClosed) return;
      const source = context.createMediaStreamSource(new MediaStream([track]));
      source.connect(mix);
      roomSources.push(source);
    },
    mute: stopMicrophone,
    finish() {
      stopMicrophone();
      const deadline = Date.now() + MAX_DRAIN_MS;
      function closeWhenPlayed() {
        if (isClosed) return;
        if (bufferedS > DRAINED_S && Date.now() < deadline) {
          setTimeout(closeWhenPlayed, DRAIN_POLL_MS);
          return;
        }
        close();
      }
      setTimeout(closeWhenPlayed, DRAIN_POLL_MS);
    },
    close,
  };
}
