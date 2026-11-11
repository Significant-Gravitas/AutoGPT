// The call's audio in the page, beside its room: the context and the player
// the click starts, the microphone, and the levels mix the call's aura moves
// with. The room publishes the microphone and plays Otto through the player
// (callRoom).

// Otto's track counts as carrying sound once a sample in it reaches this
// level (about -34 dBFS); it is read this often until it does.
const SOUND_LEVEL = 0.02;
const SOUND_POLL_MS = 50;

export interface CallAudio {
  // The person's voice and Otto's mixed, for the call's aura.
  levels: MediaStream;
  // Mixes Otto's speech from the call's room into the levels, telling
  // *onSound* once, when it first carries sound.
  listen(track: MediaStreamTrack, onSound?: () => void): void;
  // Releases the microphone now.
  mute(): void;
  // Releases the microphone and closes the context.
  close(): void;
}

// Browsers only let a page start audio from a gesture, so the context is made
// inside the click that starts or rejoins the call, before anything awaits.
export function createCallContext() {
  const context = new AudioContext({ latencyHint: "interactive" });
  void context.resume().catch(() => undefined);
  return context;
}

// Otto's speech plays through an audio element, which a browser lets play
// only once a gesture has started it: it is made, and started on silence,
// inside the same click.
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

export function createCallAudio(
  context: AudioContext,
  microphone: MediaStream,
): CallAudio {
  const source = context.createMediaStreamSource(microphone);
  const mix = context.createMediaStreamDestination();
  source.connect(mix);
  const roomSources: MediaStreamAudioSourceNode[] = [];
  const soundWatches = new Set<ReturnType<typeof setInterval>>();
  let isMicrophoneOpen = true;
  let isClosed = false;

  function mute() {
    if (!isMicrophoneOpen) return;
    isMicrophoneOpen = false;
    microphone.getTracks().forEach((track) => track.stop());
    source.disconnect();
  }

  function watchForSound(otto: AudioNode, onSound: () => void) {
    const analyser = context.createAnalyser();
    otto.connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    const watch = setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      if (!samples.some((sample) => Math.abs(sample) >= SOUND_LEVEL)) return;
      clearInterval(watch);
      soundWatches.delete(watch);
      onSound();
    }, SOUND_POLL_MS);
    soundWatches.add(watch);
  }

  return {
    levels: mix.stream,
    listen(track, onSound) {
      if (isClosed) return;
      const otto = context.createMediaStreamSource(new MediaStream([track]));
      otto.connect(mix);
      roomSources.push(otto);
      if (onSound) watchForSound(otto, onSound);
    },
    mute,
    close() {
      if (isClosed) return;
      isClosed = true;
      mute();
      soundWatches.forEach((watch) => clearInterval(watch));
      soundWatches.clear();
      roomSources.forEach((otto) => otto.disconnect());
      void context.close().catch(() => undefined);
    },
  };
}
