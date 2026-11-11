import { type Mock, vi } from "vitest";

// The browser APIs the onboarding call screen runs on, as test doubles: the
// relay's socket, the audio context, the microphone and the call's LiveKit
// room. Sockets to anywhere but the relay keep going to the socket MSW
// serves.

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
  readonly sent: unknown[] = [];

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

  send(data: unknown) {
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

  // A binary frame, which the relay never sends.
  relayBinary(bytes = 960) {
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
}

class FakeAudioNode {
  connect<T>(target: T): T {
    return target;
  }

  disconnect() {}
}

let audioContexts: FakeAudioContext[] = [];
// The level every analyser reads off the audio it is fed: silence, until a
// test has Otto speak.
let soundLevel = 0;

export class FakeAudioContext {
  state: AudioContextState = "running";
  sampleRate = 48_000;
  destination = new FakeAudioNode();
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
    // A stream an audio element takes as its source.
    return Object.assign(new FakeAudioNode(), { stream: new MediaStream() });
  }

  createAnalyser() {
    return Object.assign(new FakeAudioNode(), {
      fftSize: 2048,
      smoothingTimeConstant: 0.8,
      frequencyBinCount: 1024,
      getByteFrequencyData: (_samples: Uint8Array) => undefined,
      getFloatTimeDomainData: (samples: Float32Array) =>
        samples.fill(soundLevel),
    });
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

// --- LiveKit -----------------------------------------------------------------

// The names and values `livekit-client` gives them, as the page reads them.
const ROOM_EVENTS = {
  TrackSubscribed: "trackSubscribed",
  Reconnecting: "reconnecting",
  Reconnected: "reconnected",
  Disconnected: "disconnected",
  ParticipantDisconnected: "participantDisconnected",
} as const;
export const DISCONNECT_REASON = {
  CLIENT_INITIATED: 1,
  DUPLICATE_IDENTITY: 2,
  PARTICIPANT_REMOVED: 4,
  ROOM_DELETED: 5,
  SIGNAL_CLOSE: 9,
} as const;

let liveKitRooms: FakeRoom[] = [];
let roomJoin: { refuse: unknown } | "hang" | null = null;
// While set, joins wait on it before they answer.
let roomJoinGate: Promise<void> | null = null;
let isBrowserSupported = true;
// What WebRTC's stats say of Otto's speech, by stats id, and how often the
// page has read them; no report until a test sets one.
let ottoStatsReport: Map<string, Record<string, unknown>> | undefined;
let ottoStatsReads = 0;

// Otto's speech in a call's room.
export class FakeRemoteTrack {
  readonly kind = "audio";
  readonly mediaStreamTrack = {
    kind: "audio",
    id: "otto-track",
  } as unknown as MediaStreamTrack;
  readonly attachedTo: HTMLMediaElement[] = [];

  attach(element: HTMLMediaElement) {
    this.attachedTo.push(element);
    return element;
  }

  async getRTCStatsReport() {
    ottoStatsReads += 1;
    return ottoStatsReport as RTCStatsReport | undefined;
  }
}

// A call's LiveKit room as `livekit-client` shows it to the page.
export class FakeRoom {
  url: string | null = null;
  token: string | null = null;
  // The tracks the page published, with the options it published each
  // with, and those it took back.
  readonly published: unknown[] = [];
  readonly publishOptions: unknown[] = [];
  readonly unpublished: unknown[] = [];
  isLeft = false;
  readonly remoteParticipants = new Map<string, { identity: string }>();
  readonly localParticipant = {
    publishTrack: async (track: unknown, options?: unknown) => {
      this.published.push(track);
      this.publishOptions.push(options);
      return {};
    },
    unpublishTrack: async (track: unknown, _stopOnUnpublish?: boolean) => {
      this.unpublished.push(track);
      return undefined;
    },
  };
  private readonly handlers = new Map<string, ((...args: never[]) => void)[]>();

  constructor() {
    liveKitRooms.push(this);
  }

  on(event: string, handler: (...args: never[]) => void) {
    this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    return this;
  }

  async connect(url: string, token: string) {
    this.url = url;
    this.token = token;
    if (roomJoinGate) await roomJoinGate;
    if (roomJoin === "hang") await new Promise(() => undefined);
    if (roomJoin && roomJoin !== "hang") throw roomJoin.refuse;
  }

  async disconnect(_stopTracks?: boolean) {
    if (this.isLeft) return;
    this.isLeft = true;
    this.emit(ROOM_EVENTS.Disconnected, DISCONNECT_REASON.CLIENT_INITIATED);
  }

  // --- LiveKit's side ---

  // The relay joins as Otto, and its speech reaches the page.
  ottoJoins() {
    const otto = { identity: "otto" };
    this.remoteParticipants.set(otto.identity, otto);
    const track = new FakeRemoteTrack();
    this.emit(ROOM_EVENTS.TrackSubscribed, track, {}, otto);
    return track;
  }

  ottoLeaves() {
    const otto = this.remoteParticipants.get("otto");
    this.remoteParticipants.delete("otto");
    this.emit(ROOM_EVENTS.ParticipantDisconnected, otto);
  }

  // A track of *kind* from *identity* reaches the page.
  trackFrom(identity: string, kind = "audio") {
    const track = Object.assign(new FakeRemoteTrack(), { kind });
    this.emit(ROOM_EVENTS.TrackSubscribed, track, {}, { identity });
    return track;
  }

  reconnecting() {
    this.emit(ROOM_EVENTS.Reconnecting);
  }

  reconnected() {
    this.emit(ROOM_EVENTS.Reconnected);
  }

  // LiveKit gives the room up, for *reason*.
  lose(reason: number = DISCONNECT_REASON.SIGNAL_CLOSE) {
    this.emit(ROOM_EVENTS.Disconnected, reason);
  }

  private emit(event: string, ...args: unknown[]) {
    (this.handlers.get(event) ?? []).forEach((handler) =>
      (handler as (...values: unknown[]) => void)(...args),
    );
  }
}

// What the page imports from `livekit-client`, as fakes.
export const fakeLiveKit = {
  Room: FakeRoom,
  RoomEvent: ROOM_EVENTS,
  DisconnectReason: DISCONNECT_REASON,
  Track: { Kind: { Audio: "audio" }, Source: { Microphone: "microphone" } },
  isBrowserSupported: () => isBrowserSupported,
};

export interface VoiceFakes {
  sockets: FakeWebSocket[];
  getUserMedia: Mock<() => Promise<MediaStream>>;
  // The relay socket the page opened last.
  socket(): FakeWebSocket | undefined;
  contexts(): FakeAudioContext[];
  // Otto's speech in the call's room carries sound from now on.
  ottoSounds(): void;
  // WebRTC's stats for Otto's speech say *stats* from now on; how often the
  // page has read them.
  ottoStats(stats: Record<string, unknown>[]): void;
  ottoStatsReads(): number;
  // The next ask for the microphone fails with this error name. A refusal
  // (NotAllowedError) leaves the permission denied, as a browser does.
  refuseMicrophone(name: string): void;
  // The browser denied the microphone before the page asked.
  denyMicrophonePermission(): void;
  // LiveKit says this browser cannot join a room.
  unsupportBrowser(): void;
  // The LiveKit rooms the page made, and the one it made last.
  rooms(): FakeRoom[];
  room(): FakeRoom | undefined;
  // The next joins of a call's room are refused with *error*, or never
  // answer; `joinRooms` lets them through again.
  refuseRoomJoins(error: unknown): void;
  hangRoomJoins(): void;
  joinRooms(): void;
  // The next joins wait until the function this returns lets them in.
  holdRoomJoins(): () => void;
  uninstall(): void;
}

export function installVoiceFakes(): VoiceFakes {
  const sockets: FakeWebSocket[] = [];
  relaySockets = sockets;
  audioContexts = [];
  soundLevel = 0;
  ottoStatsReport = undefined;
  ottoStatsReads = 0;
  liveKitRooms = [];
  roomJoin = null;
  roomJoinGate = null;
  isBrowserSupported = true;
  PassThroughWebSocket = globalThis.WebSocket;
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("AudioContext", FakeAudioContext);

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

  return {
    sockets,
    getUserMedia,
    socket: () => sockets.at(-1),
    contexts: () => audioContexts,
    ottoSounds() {
      soundLevel = 0.2;
    },
    ottoStats(stats) {
      ottoStatsReport = new Map(stats.map((stat) => [String(stat.id), stat]));
    },
    ottoStatsReads: () => ottoStatsReads,
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
    unsupportBrowser() {
      isBrowserSupported = false;
    },
    rooms: () => liveKitRooms,
    room: () => liveKitRooms.at(-1),
    refuseRoomJoins(error) {
      roomJoin = { refuse: error };
    },
    hangRoomJoins() {
      roomJoin = "hang";
    },
    joinRooms() {
      roomJoin = null;
    },
    holdRoomJoins() {
      let letIn = () => undefined as void;
      roomJoinGate = new Promise<void>((resolve) => {
        letIn = resolve;
      });
      return () => {
        roomJoinGate = null;
        letIn();
      };
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
