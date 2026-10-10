import { afterEach, expect, it, vi } from "vitest";
import { requestNativePush } from "./bridge";

afterEach(() => {
  delete window.AutoGPTPush;
  delete window.webkit;
  vi.useRealTimers();
});

it("matches Android replies to the active request and rejects malformed statuses", async () => {
  const send = vi.fn();
  window.AutoGPTPush = { postMessage: send };
  let resolved = false;
  const pending = requestNativePush("status", "current-user").then((value) => {
    resolved = true;
    return value;
  });
  const sent = JSON.parse(send.mock.calls[0][0]);
  function reply(data: object) {
    window.AutoGPTPush?.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }
  reply({ id: "another-request", permission: "granted" });
  reply({ id: sent.id, permission: "unknown" });
  await Promise.resolve();
  expect(resolved).toBe(false);
  reply({ id: sent.id, permission: "disabled" });
  expect(await pending).toEqual({ id: sent.id, permission: "disabled" });
  expect(sent.account_id).toBe("current-user");
});

it("accepts the iOS message handler without assuming the browser Push API exists", async () => {
  const send = vi.fn();
  window.webkit = { messageHandlers: { AutoGPTPush: { postMessage: send } } };
  const pending = requestNativePush("enable", "current-user");
  const { id } = JSON.parse(send.mock.calls[0][0]);
  window.dispatchEvent(
    new CustomEvent("autogpt-native-push", {
      detail: { id, permission: "denied" },
    }),
  );
  expect((await pending).permission).toBe("denied");
});

it("recovers from a bridge that never replies", async () => {
  vi.useFakeTimers();
  window.AutoGPTPush = { postMessage: vi.fn() };
  const pending = expect(
    requestNativePush("enable", "current-user"),
  ).rejects.toThrow("timed out");
  await vi.advanceTimersByTimeAsync(20_000);
  await pending;
});
