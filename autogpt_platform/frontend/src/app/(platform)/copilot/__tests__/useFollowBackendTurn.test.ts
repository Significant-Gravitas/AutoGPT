import { act, renderHook } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useFollowBackendTurn } from "../useFollowBackendTurn";

function setup(status: string) {
  const hasResumedRef = { current: true };
  const refetchSession = vi.fn(async () => ({}));
  const hook = renderHook(
    ({ status }) =>
      useFollowBackendTurn({ status, refetchSession, hasResumedRef }),
    { initialProps: { status } },
  );
  return { hook, hasResumedRef, refetchSession };
}

test("an idle chat re-arms the resume and refetches at once", () => {
  const { hook, hasResumedRef, refetchSession } = setup("ready");

  act(() => hook.result.current.followBackendTurn());

  expect(hasResumedRef.current).toBe(false);
  expect(refetchSession).toHaveBeenCalledTimes(1);
});

test("a running turn is followed only once it ends", () => {
  const { hook, hasResumedRef, refetchSession } = setup("streaming");

  act(() => hook.result.current.followBackendTurn());
  expect(refetchSession).not.toHaveBeenCalled();
  expect(hasResumedRef.current).toBe(true);

  hook.rerender({ status: "ready" });

  expect(refetchSession).toHaveBeenCalledTimes(1);
  expect(hasResumedRef.current).toBe(false);
});
