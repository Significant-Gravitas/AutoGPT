import { type Mock, vi } from "vitest";

// The browser APIs the onboarding call screen runs on, as test doubles: the
// relay's socket, the audio context and its worklets, and the microphone.
// Sockets to anywhere but the relay keep going to the socket MSW serves.

export const RELAY_ORIGIN = "wss://relay.test";
export const RELAY_URL = `${RELAY_ORIGIN}/voice/call`;

type Handler<E> = ((event: E) => void) | null;

let relaySockets: FakeWebSocket[] | null = null;
let PassThroughWebSocket: typeof WebSocket | null = null;

export class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  url = "";
  readyState = FakeWebSocket.CONNECTING;
  binaryType: BinaryType = "blob";
  onopen: Handler<Event> = null;
  onmessage: Handler<MessageEvent> = null;
  onclose: Handler<CloseEvent> = null;
  onerror: Handler<Event> = null;
  readonly sent: (string | ArrayBuffer)[] = [];

  constructor(url: string, protocols?: string | string[]) {
    if (!url.startsWith(RELAY_ORIGIN) && PassThroughWebSocket) {
      return new PassThroughWebSocket(
        url,
        protocols,
      ) as unknown as FakeWebSocket;
    }
    this.url = url;
    relaySockets?.push(this);
  }

  send(data: string | ArrayBuffer) {
    if (this.readyState !== FakeWebSocket.OPEN) {
      throw new Error("The socket is not open");
    }
    this.sent.push(data);
  }

  close(code = 1000) {
    if (this.readyState >= FakeWebSocket.CLOSING) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code } as CloseEvent);
  }

  // --- The relay's side ---

  accept() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  relay(message: Record<string, unknown>) {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }

  relayAudio(bytes = 960) {
    this.onmessage?.({ data: new ArrayBuffer(bytes) } as MessageEvent);
  }

  // The connection goes, as a network drop or a relay's refusal closes it.
  drop(code = 1006) {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code } as CloseEvent);
  }

  // What the page sent as JSON, in order.
  messages() {
    return this.sent
      .filter((data): data is string => typeof data === "string")
      .map((data) => JSON.parse(data) as Record<string, unknown>);
  }

  // The audio frames the page sent.
  frames() {
    return this.sent.filter(
      (data): data is ArrayBuffer => data instanceof ArrayBuffer,
    );
  }
}

class FakeAudioNode {
  connect<T>(target: T): T {
    return target;
  }

  disconnect() {}
}

let audioContexts: FakeAudioContext[] = [];
let workletNodes: FakeAudioWorkletNode[] = [];
let workletFailure: Error | null = null;

export class FakeAudioContext {
  state: AudioContextState = "running";
  sampleRate = 48_000;
  destination = new FakeAudioNode();
  audioWorklet = {
    addModule: vi.fn(async (_url: string) => {
      if (workletFailure) throw workletFailure;
    }),
  };
  resume = vi.fn(async () => {
    this.state = "running";
  });
  close = vi.fn(async () => {
    this.state = "closed";
  });

  constructor(_options?: AudioContextOptions) {
    audioContexts.push(this);
  }

  createMediaStreamSource(_stream: MediaStream) {
    return new FakeAudioNode();
  }

  createGain() {
    return Object.assign(new FakeAudioNode(), { gain: { value: 1 } });
  }

  createMediaStreamDestination() {
    return Object.assign(new FakeAudioNode(), { stream: fakeMediaStream() });
  }

  createAnalyser() {
    return Object.assign(new FakeAudioNode(), {
      fftSize: 2048,
      smoothingTimeConstant: 0.8,
      frequencyBinCount: 1024,
      getByteFrequencyData: (_samples: Uint8Array) => undefined,
    });
  }
}

export class FakeAudioWorkletNode extends FakeAudioNode {
  readonly port = {
    onmessage: null as Handler<MessageEvent>,
    postMessage: vi.fn(),
  };

  constructor(
    readonly context: FakeAudioContext,
    readonly name: string,
    readonly options?: AudioWorkletNodeOptions,
  ) {
    super();
    workletNodes.push(this);
  }
}

export function fakeMediaStream() {
  const track = { kind: "audio", stop: vi.fn() };
  return {
    id: "fake-stream",
    active: true,
    getTracks: () => [track],
    getAudioTracks: () => [track],
  } as unknown as MediaStream;
}

export interface VoiceFakes {
  sockets: FakeWebSocket[];
  getUserMedia: Mock<() => Promise<MediaStream>>;
  // The relay socket the page opened last.
  socket(): FakeWebSocket | undefined;
  contexts(): FakeAudioContext[];
  // The capture and playback worklets of the call last connected.
  micNode(): FakeAudioWorkletNode | undefined;
  playerNode(): FakeAudioWorkletNode | undefined;
  // The microphone posts one 40 ms frame of 16 kHz PCM16.
  speak(): void;
  // The next ask for the microphone fails with this error name. A refusal
  // (NotAllowedError) leaves the permission denied, as a browser does.
  refuseMicrophone(name: string): void;
  // The browser denied the microphone before the page asked.
  denyMicrophonePermission(): void;
  // The audio worklets fail to load, as on a browser that cannot run them.
  failWorklets(): void;
  uninstall(): void;
}

export function installVoiceFakes(): VoiceFakes {
  const sockets: FakeWebSocket[] = [];
  relaySockets = sockets;
  audioContexts = [];
  workletNodes = [];
  workletFailure = null;
  PassThroughWebSocket = globalThis.WebSocket;
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("AudioContext", FakeAudioContext);
  vi.stubGlobal("AudioWorkletNode", FakeAudioWorkletNode);

  // The microphone's permission, as the Permissions API answers it: asked
  // for on the first call, as in a browser that has not been asked yet.
  let permission: PermissionState = "prompt";
  const getUserMedia = vi.fn(async () => {
    permission = "granted";
    return fakeMediaStream();
  });
  const mediaDevices = Object.getOwnPropertyDescriptor(
    navigator,
    "mediaDevices",
  );
  const permissions = Object.getOwnPropertyDescriptor(navigator, "permissions");
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  Object.defineProperty(navigator, "permissions", {
    configurable: true,
    value: {
      query: async () => ({
        state: permission,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
      }),
    },
  });

  function lastNode(name: string) {
    return workletNodes.filter((node) => node.name === name).at(-1);
  }

  return {
    sockets,
    getUserMedia,
    socket: () => sockets.at(-1),
    contexts: () => audioContexts,
    micNode: () => lastNode("mic-capture"),
    playerNode: () => lastNode("pcm-player"),
    speak() {
      lastNode("mic-capture")?.port.onmessage?.({
        data: new Int16Array(640).buffer,
      } as MessageEvent);
    },
    refuseMicrophone(name) {
      getUserMedia.mockImplementation(async () => {
        if (name === "NotAllowedError") permission = "denied";
        throw Object.assign(new Error("The microphone was refused"), {
          name,
        });
      });
    },
    denyMicrophonePermission() {
      permission = "denied";
    },
    failWorklets() {
      workletFailure = new Error("The audio worklet could not be loaded");
    },
    uninstall() {
      vi.unstubAllGlobals();
      restore("mediaDevices", mediaDevices);
      restore("permissions", permissions);
      relaySockets = null;
      PassThroughWebSocket = null;
    },
  };
}

function restore(
  key: "mediaDevices" | "permissions",
  was?: PropertyDescriptor,
) {
  if (was) {
    Object.defineProperty(navigator, key, was);
    return;
  }
  delete (navigator as unknown as Record<string, unknown>)[key];
}
