import { act, renderHook } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { useFollowBackendTurn } from "../useFollowBackendTurn";

const idle = { data: { status: 200, data: { active_stream: null } } };
const running = {
  data: { status: 200, data: { active_stream: { turn_id: "t" } } },
};

afterEach(() => {
  vi.useRealTimers();
});

function setup(status: string, results: { data?: unknown }[] = [running]) {
  const hasResumedRef = { current: true };
  const refetchSession = vi.fn(async () => results.shift() ?? idle);
  const hook = renderHook(
    ({ status }) =>
      useFollowBackendTurn({ status, refetchSession, hasResumedRef }),
    { initialProps: { status } },
  );
  return { hook, hasResumedRef, refetchSession };
}

test("an idle chat re-arms the resume and refetches at once", async () => {
  const { hook, hasResumedRef, refetchSession } = setup("ready");

  await act(async () => hook.result.current.followBackendTurn());

  expect(hasResumedRef.current).toBe(false);
  expect(refetchSession).toHaveBeenCalledTimes(1);
});

test("a running turn is followed only once it ends", async () => {
  const { hook, hasResumedRef, refetchSession } = setup("streaming");

  await act(async () => hook.result.current.followBackendTurn());
  expect(refetchSession).not.toHaveBeenCalled();
  expect(hasResumedRef.current).toBe(true);

  await act(async () => hook.rerender({ status: "ready" }));

  expect(refetchSession).toHaveBeenCalledTimes(1);
  expect(hasResumedRef.current).toBe(false);
});

test("a turn that ended while the answer was posting is still followed", async () => {
  const { hook, refetchSession } = setup("streaming");
  const clickedWhileStreaming = hook.result.current.followBackendTurn;

  await act(async () => hook.rerender({ status: "ready" }));
  await act(async () => clickedWhileStreaming());

  expect(refetchSession).toHaveBeenCalledTimes(1);
});

test("the probe retries until the answer's turn has a stream", async () => {
  vi.useFakeTimers();
  const { hook, refetchSession } = setup("ready", [idle, idle, running]);

  await act(async () => hook.result.current.followBackendTurn());
  await act(async () => vi.advanceTimersByTimeAsync(5_000));

  expect(refetchSession).toHaveBeenCalledTimes(3);
});

test("a second answer mid-probe does not start a second probe", async () => {
  vi.useFakeTimers();
  let inFlight = 0;
  let maxInFlight = 0;
  const hasResumedRef = { current: true };
  async function refetchSession() {
    maxInFlight = Math.max(maxInFlight, ++inFlight);
    await new Promise((r) => setTimeout(r, 200));
    inFlight--;
    return idle;
  }
  const hook = renderHook(() =>
    useFollowBackendTurn({ status: "ready", refetchSession, hasResumedRef }),
  );

  await act(async () => hook.result.current.followBackendTurn());
  await act(async () => hook.result.current.followBackendTurn());
  await act(async () => vi.advanceTimersByTimeAsync(10_000));

  expect(maxInFlight).toBe(1);
});
