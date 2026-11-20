import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useSettleGate } from "./useSettleGate";

interface Props {
  isSettled: boolean;
  scope?: string;
}

function renderGate(initialProps: Props) {
  return renderHook(
    ({ isSettled, scope }: Props) => useSettleGate(isSettled, scope),
    { initialProps },
  );
}

describe("useSettleGate", () => {
  it("releases a waiter once its chat settles", async () => {
    const { result, rerender } = renderGate({
      isSettled: false,
      scope: "chat-1",
    });
    const wait = result.current.waitUntilSettled();

    rerender({ isSettled: true, scope: "chat-1" });

    await expect(wait).resolves.toBe(true);
  });

  it("tells a waiter its chat was left, even when the next one has settled", async () => {
    const { result, rerender } = renderGate({
      isSettled: false,
      scope: "chat-1",
    });
    const wait = result.current.waitUntilSettled();

    rerender({ isSettled: true, scope: "chat-2" });

    await expect(wait).resolves.toBe(false);
  });

  it("tells a waiter the host went away when it unmounts", async () => {
    const { result, unmount } = renderGate({ isSettled: false });
    const wait = result.current.waitUntilSettled();

    unmount();

    await expect(wait).resolves.toBe(false);
  });
});
