import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_WAVY_ORB_SETTINGS } from "../WavyOrb/helpers";
import { WavyOrb } from "../WavyOrb/WavyOrb";

const { reducedMotion } = vi.hoisted(() => ({
  reducedMotion: { value: false },
}));

vi.mock("framer-motion", async (importOriginal) => {
  const original = await importOriginal<typeof import("framer-motion")>();
  return {
    ...original,
    useReducedMotion: () => reducedMotion.value,
  };
});

type ObserverCallback = (entries: Array<{ isIntersecting: boolean }>) => void;

const animationFrames = new Map<number, FrameRequestCallback>();
const intersectionCallbacks: ObserverCallback[] = [];
const resizeCallbacks: Array<() => void> = [];
const closeAudioContext = vi.fn(() => Promise.resolve());
const resumeAudioContext = vi.fn(() => Promise.resolve());
let animationFrameId = 0;

function createWebGlContext() {
  const shader = {} as WebGLShader;
  const program = {} as WebGLProgram;
  const vertexArray = {} as WebGLVertexArrayObject;

  return {
    VERTEX_SHADER: 35633,
    FRAGMENT_SHADER: 35632,
    COMPILE_STATUS: 35713,
    LINK_STATUS: 35714,
    TRIANGLES: 4,
    createShader: vi.fn(() => shader),
    shaderSource: vi.fn(),
    compileShader: vi.fn(),
    getShaderParameter: vi.fn(() => true),
    deleteShader: vi.fn(),
    createProgram: vi.fn(() => program),
    attachShader: vi.fn(),
    linkProgram: vi.fn(),
    getProgramParameter: vi.fn(() => true),
    deleteProgram: vi.fn(),
    createVertexArray: vi.fn(() => vertexArray),
    bindVertexArray: vi.fn(),
    deleteVertexArray: vi.fn(),
    useProgram: vi.fn(),
    getUniformLocation: vi.fn(
      (_program: WebGLProgram, name: string) =>
        ({ name }) as unknown as WebGLUniformLocation,
    ),
    viewport: vi.fn(),
    uniform2f: vi.fn(),
    uniform1f: vi.fn(),
    uniform3f: vi.fn(),
    drawArrays: vi.fn(),
  };
}

function flushAnimationFrames(count = 1) {
  for (let index = 0; index < count; index++) {
    const pending = Array.from(animationFrames.values());
    animationFrames.clear();
    for (const callback of pending) callback(performance.now());
  }
}

function setReducedMotion(matches: boolean) {
  reducedMotion.value = matches;
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockReturnValue({
      matches,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
    }),
  );
}

beforeEach(() => {
  animationFrames.clear();
  intersectionCallbacks.length = 0;
  resizeCallbacks.length = 0;
  animationFrameId = 0;
  closeAudioContext.mockClear();
  resumeAudioContext.mockClear();
  setReducedMotion(false);

  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    animationFrames.set(++animationFrameId, callback);
    return animationFrameId;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    animationFrames.delete(id);
  });
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      constructor(callback: ObserverCallback) {
        intersectionCallbacks.push(callback);
      }
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: () => void) {
        resizeCallbacks.push(callback);
      }
      observe() {}
      disconnect() {}
    },
  );
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get: () => 620,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => 240,
  });
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: "visible",
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  Reflect.deleteProperty(HTMLElement.prototype, "clientWidth");
  Reflect.deleteProperty(HTMLElement.prototype, "clientHeight");
});

describe("WavyOrb", () => {
  it("draws while visible and releases WebGL resources on unmount", () => {
    const gl = createWebGlContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      gl as unknown as WebGL2RenderingContext,
    );

    const { unmount } = render(
      <WavyOrb audioStream={null} settings={DEFAULT_WAVY_ORB_SETTINGS} />,
    );

    expect(screen.getByTestId("orb-wavy")).toBeDefined();
    expect(gl.createShader).toHaveBeenCalledTimes(2);

    act(() => {
      flushAnimationFrames(2);
      resizeCallbacks[0]();
      intersectionCallbacks[0]([{ isIntersecting: false }]);
    });
    expect(gl.drawArrays).toHaveBeenCalled();
    expect(animationFrames.size).toBe(0);

    act(() => {
      intersectionCallbacks[0]([{ isIntersecting: true }]);
      flushAnimationFrames();
    });
    expect(gl.viewport).toHaveBeenCalledWith(0, 0, 620, 240);

    unmount();
    expect(gl.deleteVertexArray).toHaveBeenCalled();
    expect(gl.deleteProgram).toHaveBeenCalled();
  });

  it("uses microphone frequency bands when a stream is available", () => {
    const gl = createWebGlContext();
    const getByteFrequencyData = vi.fn((values: Uint8Array) => values.fill(64));
    const connect = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      gl as unknown as WebGL2RenderingContext,
    );
    vi.stubGlobal(
      "AudioContext",
      class {
        state: AudioContextState = "suspended";
        sampleRate = 48000;
        createAnalyser() {
          return {
            fftSize: 1024,
            frequencyBinCount: 512,
            smoothingTimeConstant: 0,
            getByteFrequencyData,
          };
        }
        createMediaStreamSource() {
          return { connect };
        }
        close = closeAudioContext;
        resume = resumeAudioContext;
      },
    );

    const { unmount } = render(
      <WavyOrb
        audioStream={{} as MediaStream}
        settings={DEFAULT_WAVY_ORB_SETTINGS}
      />,
    );

    act(() => flushAnimationFrames());
    expect(connect).toHaveBeenCalled();
    expect(resumeAudioContext).toHaveBeenCalled();
    expect(getByteFrequencyData).toHaveBeenCalled();
    expect(gl.uniform1f).toHaveBeenCalled();

    unmount();
    expect(closeAudioContext).toHaveBeenCalled();
  });

  it("draws one static frame when reduced motion is requested", () => {
    setReducedMotion(true);
    const gl = createWebGlContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      gl as unknown as WebGL2RenderingContext,
    );

    render(<WavyOrb audioStream={null} settings={DEFAULT_WAVY_ORB_SETTINGS} />);

    expect(gl.drawArrays).toHaveBeenCalledTimes(1);
    expect(animationFrames.size).toBe(0);
    expect(gl.uniform1f).toHaveBeenCalledWith(
      expect.objectContaining({ name: "uTime" }),
      3.2,
    );
  });

  it("renders a fallback when WebGL2 is unavailable", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);

    render(<WavyOrb audioStream={null} settings={DEFAULT_WAVY_ORB_SETTINGS} />);

    expect(screen.getByTestId("orb-wavy").tagName).toBe("DIV");
  });

  it("renders a fallback when shader compilation fails", () => {
    const gl = createWebGlContext();
    gl.getShaderParameter.mockReturnValue(false);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      gl as unknown as WebGL2RenderingContext,
    );

    render(<WavyOrb audioStream={null} settings={DEFAULT_WAVY_ORB_SETTINGS} />);

    expect(screen.getByTestId("orb-wavy").tagName).toBe("DIV");
    expect(gl.deleteShader).toHaveBeenCalled();
  });
});
