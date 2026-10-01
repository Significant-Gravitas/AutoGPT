import { act, renderHook } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it } from "vitest";

import { useLocalStreamSettle } from "../useLocalStreamSettle";

interface Props {
  sessionId: string | null;
  isSettled: boolean;
}

function setup(initial: Props) {
  return renderHook((props: Props) => useLocalStreamSettle(props), {
    initialProps: initial,
  });
}

async function settleMicrotasks() {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("useLocalStreamSettle", () => {
  it("resolves right away when the stream is already settled", async () => {
    const view = setup({ sessionId: "s1", isSettled: true });
    await expect(view.result.current.waitForLocalSettle("s1")).resolves.toBe(
      true,
    );
  });

  it("holds until the stream settles, then resolves true", async () => {
    const view = setup({ sessionId: "s1", isSettled: false });
    let outcome: boolean | null = null;
    void view.result.current
      .waitForLocalSettle("s1")
      .then((value) => (outcome = value));

    await settleMicrotasks();
    expect(outcome).toBeNull();

    view.rerender({ sessionId: "s1", isSettled: true });
    await settleMicrotasks();
    expect(outcome).toBe(true);
  });

  it("resolves false when the chat changes before the stream settles", async () => {
    const view = setup({ sessionId: "s1", isSettled: false });
    let outcome: boolean | null = null;
    void view.result.current
      .waitForLocalSettle("s1")
      .then((value) => (outcome = value));

    view.rerender({ sessionId: "s2", isSettled: false });
    await settleMicrotasks();
    expect(outcome).toBe(false);

    // The new chat settling must not revive the old chat's waiter.
    view.rerender({ sessionId: "s2", isSettled: true });
    await settleMicrotasks();
    expect(outcome).toBe(false);
  });

  it("resolves false when asked to wait for a chat that is no longer current", async () => {
    const view = setup({ sessionId: "s2", isSettled: true });
    await expect(view.result.current.waitForLocalSettle("s1")).resolves.toBe(
      false,
    );
  });

  it("resolves false on unmount before the stream settles", async () => {
    const view = setup({ sessionId: "s1", isSettled: false });
    let outcome: boolean | null = null;
    void view.result.current
      .waitForLocalSettle("s1")
      .then((value) => (outcome = value));

    view.unmount();
    await settleMicrotasks();
    expect(outcome).toBe(false);
  });

  it("resolves false when asked to wait after unmount, settled or not", async () => {
    const settled = setup({ sessionId: "s1", isSettled: true });
    settled.unmount();
    await expect(settled.result.current.waitForLocalSettle("s1")).resolves.toBe(
      false,
    );

    const unsettled = setup({ sessionId: "s1", isSettled: false });
    unsettled.unmount();
    await expect(
      unsettled.result.current.waitForLocalSettle("s1"),
    ).resolves.toBe(false);
  });

  it("still waits and settles after StrictMode's simulated remount", async () => {
    const view = renderHook((props: Props) => useLocalStreamSettle(props), {
      initialProps: { sessionId: "s1", isSettled: false },
      wrapper: StrictMode,
    });
    let outcome: boolean | null = null;
    void view.result.current
      .waitForLocalSettle("s1")
      .then((value) => (outcome = value));

    await settleMicrotasks();
    expect(outcome).toBeNull();

    view.rerender({ sessionId: "s1", isSettled: true });
    await settleMicrotasks();
    expect(outcome).toBe(true);
  });
});
