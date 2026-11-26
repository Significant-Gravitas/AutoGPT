import { afterEach, describe, expect, test, vi } from "vitest";
import { createRouterQueue, disposeRouterQueue } from "../next-router-queue";

// patches/next@15.5.28.patch backports both halves of vercel/next.js#95391's
// router queue fix (#16965). These drive Next's installed queue, so they fail
// on an unpatched 15.5 and keep passing once an upgrade carries the fix.

afterEach(() => {
  disposeRouterQueue();
});

function afterQueuedWork() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("Next's router queue", () => {
  test("runs a server action sent while a navigation replaces a pending one, after that navigation", async () => {
    const queue = createRouterQueue();
    const ran: string[] = [];
    let answerFirst = () => {};
    const first = queue.callServer(
      () =>
        new Promise<void>((resolve) => {
          answerFirst = resolve;
        }),
    );
    const navigation = queue.navigate();
    answerFirst();
    await first;
    const second = queue.callServer(async () => {
      ran.push("second");
    });

    await afterQueuedWork();
    expect(ran).toEqual([]);
    navigation.arrive();
    await vi.waitFor(() => expect(ran).toEqual(["second"]));
    await second;
  });

  test("doesn't start the next action while a navigation runs when the action it replaced fails", async () => {
    const queue = createRouterQueue();
    const ran: string[] = [];
    let failFirst: (error: Error) => void = () => {};
    const first = queue.callServer(
      () =>
        new Promise<void>((_resolve, reject) => {
          failFirst = reject;
        }),
      { routerFails: true },
    );
    const navigation = queue.navigate();
    const second = queue.callServer(async () => {
      ran.push("second");
    });
    failFirst(new Error("refetch failed"));
    await expect(first).rejects.toThrow("refetch failed");

    await afterQueuedWork();
    expect(ran).toEqual([]);
    navigation.arrive();
    await vi.waitFor(() => expect(ran).toEqual(["second"]));
    await second;
  });
});
