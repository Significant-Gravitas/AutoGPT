import { act, render, screen } from "@testing-library/react";
import { motionValue, type MotionValue } from "framer-motion";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MicButton, type OrbScreen } from "../MicButton";
import { VoiceDots } from "../VoiceDots";

interface Running {
  value: MotionValue<number>;
  target: number;
  options: { duration?: number; onUpdate?: (value: number) => void };
  finish: () => void;
  stop: ReturnType<typeof vi.fn>;
}
const state = vi.hoisted(() => ({
  reduced: false,
  animations: [] as Running[],
}));
vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => state.reduced,
  animate: (
    value: MotionValue<number>,
    target: number,
    options: Running["options"],
  ) => {
    let resolve = () => {};
    const promise = new Promise<void>((done) => {
      resolve = done;
    });
    const stop = vi.fn();
    state.animations.push({
      value,
      target,
      options,
      stop,
      finish() {
        value.set(target);
        options.onUpdate?.(target);
        resolve();
      },
    });
    return Object.assign(promise, { stop });
  },
}));
vi.mock("../VoiceAura", () => ({
  VoiceAura: ({
    isActive,
    children,
  }: {
    isActive: boolean;
    children: ReactNode;
  }) => (
    <div data-testid="fold-wave" data-active={isActive ? "yes" : "no"}>
      {children}
    </div>
  ),
}));
vi.mock("@/components/molecules/BotAvatar/BotAvatar", () => ({
  BotAvatar: ({
    status,
    poseOffset,
  }: {
    status: string;
    poseOffset: { pitch: number };
  }) => (
    <span
      data-testid="face"
      data-status={status}
      data-pitch={poseOffset.pitch}
    />
  ),
}));
beforeEach(() => {
  vi.useFakeTimers();
  state.reduced = false;
  state.animations = [];
});
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
});

async function overlap() {
  await act(async () => {
    vi.advanceTimersByTime(120);
  });
}
async function finish(animations: Running[]) {
  await act(async () => {
    animations.forEach((animation) => animation.finish());
  });
}

describe("microphone folding", () => {
  it.each(["rest", "processing", "failed"] as OrbScreen[])(
    "does not unfold on an initial %s screen",
    async (screenName) => {
      const { rerender } = render(
        <MicButton screen={screenName} audioStream={null} />,
      );
      await overlap();
      expect(state.animations).toHaveLength(0);
      expect(screen.getByTestId("fold-wave").getAttribute("data-active")).toBe(
        "no",
      );
      state.reduced = true;
      rerender(<MicButton screen={screenName} audioStream={null} />);
      state.reduced = false;
      rerender(<MicButton screen={screenName} audioStream={null} />);
      await overlap();
      expect(state.animations).toHaveLength(0);
    },
  );

  it("folds into listening dots and unfolds to a neutral face", async () => {
    const { rerender } = render(<MicButton screen="rest" audioStream={null} />);
    rerender(<MicButton screen="recording" audioStream={null} />);
    expect(state.animations.map((item) => item.target)).toEqual([
      Math.PI * 2,
      1,
    ]);
    expect(screen.getByTestId("fold-wave").getAttribute("data-active")).toBe(
      "yes",
    );
    await overlap();
    expect(state.animations).toHaveLength(3);
    await finish(state.animations.slice(0, 2));
    expect(screen.getByTestId("fold-wave").getAttribute("data-active")).toBe(
      "no",
    );
    await finish(state.animations.slice(2));
    expect(screen.getByTestId("fold-wave").getAttribute("data-active")).toBe(
      "no",
    );
    rerender(<MicButton screen="rest" audioStream={null} />);
    await overlap();
    expect(state.animations.slice(3).map((item) => item.target)).toEqual([
      0, 0, 0,
    ]);
    await finish(state.animations.slice(3));
    expect(screen.getByTestId("face").getAttribute("data-pitch")).toBe("0");
    expect(screen.getByTestId("fold-wave").getAttribute("data-active")).toBe(
      "no",
    );
  });

  it("cancels a partial fold without starting its delayed split", async () => {
    const { rerender, unmount } = render(
      <MicButton screen="recording" audioStream={null} />,
    );
    const folding = [...state.animations];
    act(() => folding[1].value.set(0.5));
    rerender(<MicButton screen="rest" audioStream={null} />);
    await overlap();
    expect(folding.every((item) => item.stop.mock.calls.length === 1)).toBe(
      true,
    );
    expect(state.animations.filter((item) => item.target === 1)).toHaveLength(
      1,
    );
    await finish(state.animations.slice(2));
    expect(screen.getByTestId("face").getAttribute("data-pitch")).toBe("0");
    unmount();
    expect(
      state.animations.every((item) => item.stop.mock.calls.length > 0),
    ).toBe(true);
  });

  it("uses reduced-motion transitions and resets a partially turned face", async () => {
    const { rerender } = render(
      <MicButton screen="recording" audioStream={null} />,
    );
    act(() => {
      state.animations[0].value.set(Math.PI);
      state.animations[0].options.onUpdate?.(Math.PI);
      state.animations[1].value.set(0.5);
    });
    state.reduced = true;
    rerender(<MicButton screen="rest" audioStream={null} />);
    await overlap();
    expect(
      state.animations.slice(2).every((item) => item.options.duration === 0.15),
    ).toBe(true);
    await finish(state.animations.slice(2));
    expect(screen.getByTestId("face").getAttribute("data-pitch")).toBe("0");
    rerender(<MicButton screen="recording" audioStream={null} />);
    await overlap();
    expect(
      state.animations.slice(4).every((item) => item.options.duration === 0.15),
    ).toBe(true);
    await finish(state.animations.slice(4));
    const count = state.animations.length;
    state.reduced = false;
    rerender(<MicButton screen="recording" audioStream={null} />);
    expect(state.animations).toHaveLength(count);
  });
});

it.each([false, true])(
  "voice dots respect reduced motion: %s",
  (reduceMotion) => {
    render(
      <VoiceDots
        levels={Array.from({ length: 5 }, () => motionValue(1))}
        collapse={motionValue(1)}
        split={motionValue(1)}
        centre={{ x: 80, y: 80 }}
        color="#123456"
        reduceMotion={reduceMotion}
      />,
    );
    expect(
      screen.getAllByTestId("voice-dot").map((dot) => dot.style.height),
    ).toEqual(
      reduceMotion
        ? ["14px", "14px", "14px", "14px"]
        : ["40px", "64px", "64px", "40px"],
    );
  },
);
