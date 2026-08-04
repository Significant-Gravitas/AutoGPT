import { vi } from "vitest";

// Shared fakes for the live-captions cluster. The hooks talk to four browser
// APIs that happy-dom does not provide (WebSocket streaming, AudioContext,
// ScriptProcessor, SpeechRecognition); these stand in for them and expose
// drivers so a test can decide exactly when the provider speaks.

export function fakeStream() {
  return { getTracks: () => [] } as unknown as MediaStream;
}

export class FakeWebSocket {
  static readonly OPEN = 1;
  static instances: FakeWebSocket[] = [];

  static reset() {
    FakeWebSocket.instances = [];
  }

  static last() {
    const socket = FakeWebSocket.instances.at(-1);
    if (!socket) throw new Error("no WebSocket was opened");
    return socket;
  }

  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  readyState = 0;
  sent: unknown[] = [];
  closeCount = 0;
  url: string;
  protocols?: string | string[];

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    FakeWebSocket.instances.push(this);
  }

  send(data: unknown) {
    this.sent.push(data);
  }

  close() {
    this.closeCount += 1;
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
}

export class FakeScriptProcessor {
  onaudioprocess:
    | ((event: {
        inputBuffer: { getChannelData: (channel: number) => Float32Array };
      }) => void)
    | null = null;
  connectCount = 0;
  disconnectCount = 0;

  connect() {
    this.connectCount += 1;
  }

  disconnect() {
    this.disconnectCount += 1;
  }

  feed(samples: number[]) {
    this.onaudioprocess?.({
      inputBuffer: { getChannelData: () => Float32Array.from(samples) },
    });
  }
}

export class FakeAnalyser {
  fftSize = 2048;
  frequencyBinCount = 128;
  // Distance from the 128 silence midpoint that getByteTimeDomainData reports.
  amplitude = 0;

  getByteTimeDomainData(data: Uint8Array) {
    data.fill(128 + this.amplitude);
  }
}

export class FakeAudioContext {
  static instances: FakeAudioContext[] = [];

  static reset() {
    FakeAudioContext.instances = [];
  }

  static last() {
    const context = FakeAudioContext.instances.at(-1);
    if (!context) throw new Error("no AudioContext was created");
    return context;
  }

  sampleRate: number | undefined;
  destination = {};
  closeCount = 0;
  connectedStreams: MediaStream[] = [];
  processor: FakeScriptProcessor | null = null;
  analyser: FakeAnalyser | null = null;

  constructor(options?: { sampleRate?: number }) {
    this.sampleRate = options?.sampleRate;
    FakeAudioContext.instances.push(this);
  }

  createMediaStreamSource(stream: MediaStream) {
    this.connectedStreams.push(stream);
    return { connect: () => {} };
  }

  createScriptProcessor() {
    this.processor = new FakeScriptProcessor();
    return this.processor;
  }

  createAnalyser() {
    this.analyser = new FakeAnalyser();
    return this.analyser;
  }

  close() {
    this.closeCount += 1;
    return Promise.resolve();
  }
}

export class FakeSpeechRecognition {
  static instances: FakeSpeechRecognition[] = [];

  static reset() {
    FakeSpeechRecognition.instances = [];
  }

  static last() {
    const recognition = FakeSpeechRecognition.instances.at(-1);
    if (!recognition) throw new Error("no SpeechRecognition was constructed");
    return recognition;
  }

  continuous = false;
  interimResults = false;
  startCount = 0;
  stopCount = 0;
  onresult:
    | ((event: {
        results: ArrayLike<ArrayLike<{ transcript: string }>>;
      }) => void)
    | null = null;
  onerror: ((event?: { error?: string }) => void) | null = null;
  onend: (() => void) | null = null;

  constructor() {
    FakeSpeechRecognition.instances.push(this);
  }

  start() {
    this.startCount += 1;
  }

  stop() {
    this.stopCount += 1;
  }

  say(...utterances: string[]) {
    this.onresult?.({
      results: utterances.map((transcript) => [{ transcript }]),
    });
  }
}

export function stubTokenFetch(
  response: { ok?: boolean; token?: string | undefined } = {},
) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: response.ok ?? true,
    json: async () => ({ token: "token" in response ? response.token : "tok" }),
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

// The PCM encoders clamp to [-1, 1] and pack little-endian int16.
export const PCM_SAMPLES = [1, -1, 0, 2];
export const PCM_EXPECTED = [32767, -32768, 0, 32767];
