import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createProgressSaver } from "../progress-saver";
import { readLocalProgress } from "../progress";
import { makeProgress } from "./progress-fixture";

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("onboarding draft outbox", () => {
  it("serializes writes and keeps the newest progress while an older write is pending", async () => {
    let release: ((revision: number) => void) | undefined;
    const save = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<number>((resolve) => {
            release = resolve;
          }),
      )
      .mockResolvedValue(2);
    const saver = createProgressSaver({
      initialRevision: 0,
      onConflict: vi.fn(),
      userID: "one",
      save,
      onError: vi.fn(),
    });
    saver.enqueue(makeProgress({ role: "First" }));
    const flushing = saver.flush();
    saver.enqueue(
      makeProgress({
        role: "Latest",
        currentStep: "painPoints",
        completedSteps: ["role"],
      }),
    );
    expect(save).toHaveBeenCalledTimes(1);
    expect(readLocalProgress("one")?.pending).toBe(true);
    release?.(1);
    await flushing;
    expect(save).toHaveBeenCalledTimes(2);
    expect(save.mock.calls[1][0].role).toBe("Latest");
    expect(readLocalProgress("one")).toMatchObject({
      pending: false,
      progress: { role: "Latest" },
    });
    saver.dispose();
  });

  it("keeps failed writes durable and retries without another user interaction", async () => {
    const save = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue(2);
    const onError = vi.fn();
    const saver = createProgressSaver({
      initialRevision: 0,
      onConflict: vi.fn(),
      userID: "one",
      save,
      onError,
    });
    saver.enqueue(makeProgress());
    await expect(saver.flush()).rejects.toThrow("offline");
    expect(readLocalProgress("one")?.pending).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);
    expect(save).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenLastCalledWith(null);
    expect(readLocalProgress("one")?.pending).toBe(false);
    saver.dispose();
  });

  it("aborts an old account request and leaves its draft scoped to that account", async () => {
    let release: ((revision: number) => void) | undefined;
    const save = vi.fn(
      (_progress: unknown, _revision: number, _signal: AbortSignal) =>
        new Promise<number>((resolve) => {
          release = resolve;
        }),
    );
    const saver = createProgressSaver({
      initialRevision: 0,
      onConflict: vi.fn(),
      userID: "one",
      save,
      onError: vi.fn(),
    });
    saver.enqueue(makeProgress({ role: "Private answer" }));
    const flushing = saver.flush();
    saver.dispose();
    expect(save.mock.calls[0][2].aborted).toBe(true);
    release?.(1);
    await expect(flushing).rejects.toThrow("account changed");
    await vi.advanceTimersByTimeAsync(30_000);
    expect(save).toHaveBeenCalledTimes(1);
    expect(readLocalProgress("one")?.pending).toBe(true);
    expect(readLocalProgress("two")).toBeNull();
  });
});

it("times out a hung request so checkout flush can recover", async () => {
  const save = vi.fn(
    (_progress: unknown, _revision: number, _signal: AbortSignal) =>
      new Promise<number>(() => {}),
  );
  const saver = createProgressSaver({
    initialRevision: 0,
    onConflict: vi.fn(),
    userID: "one",
    save,
    onError: vi.fn(),
  });
  saver.enqueue(makeProgress());
  const rejected = expect(saver.flush()).rejects.toThrow("timed out");
  await vi.advanceTimersByTimeAsync(10_000);
  await rejected;
  expect(save.mock.calls[0][2].aborted).toBe(true);
  expect(readLocalProgress("one")?.pending).toBe(true);
  saver.dispose();
});

it("stops automatic retries when another session has updated the server revision", async () => {
  const save = vi.fn().mockRejectedValue({ status: 409 });
  const onConflict = vi.fn();
  const saver = createProgressSaver({
    initialRevision: 3,
    userID: "one",
    save,
    onError: vi.fn(),
    onConflict,
  });
  saver.enqueue(makeProgress());
  await expect(saver.flush()).rejects.toEqual({ status: 409 });
  await vi.advanceTimersByTimeAsync(30_000);
  expect(save).toHaveBeenCalledOnce();
  expect(onConflict).toHaveBeenCalledOnce();
  expect(readLocalProgress("one")).toMatchObject({
    revision: 3,
    pending: true,
  });
  saver.dispose();
});
