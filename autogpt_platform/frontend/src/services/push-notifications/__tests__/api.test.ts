import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchVapidPublicKey,
  removeSubscriptionFromServer,
  sendSubscriptionToServer,
} from "../api";

describe("fetchVapidPublicKey", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns public key on success", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ public_key: "BFakeKey123" }),
    } as Response);

    const key = await fetchVapidPublicKey();

    expect(key).toBe("BFakeKey123");
    expect(fetch).toHaveBeenCalledWith("/api/proxy/api/push/vapid-key");
  });

  it("returns null when response is not ok", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
    } as Response);

    const key = await fetchVapidPublicKey();

    expect(key).toBeNull();
  });

  it("returns null when public_key is missing from response", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    } as Response);

    const key = await fetchVapidPublicKey();

    expect(key).toBeNull();
  });

  it("returns null on network error", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("Network error"));

    const key = await fetchVapidPublicKey();

    expect(key).toBeNull();
  });
});

describe("sendSubscriptionToServer", () => {
  const mockSubscription = {
    toJSON: () => ({
      endpoint: "https://push.example.com/sub/123",
      keys: { p256dh: "key-p256dh", auth: "key-auth" },
    }),
  } as unknown as PushSubscription;

  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends correct payload via proxy", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true } as Response);

    await sendSubscriptionToServer(mockSubscription);

    expect(fetch).toHaveBeenCalledWith("/api/proxy/api/push/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        endpoint: "https://push.example.com/sub/123",
        keys: { p256dh: "key-p256dh", auth: "key-auth" },
        user_agent: navigator.userAgent,
      }),
    });
  });

  it("returns true on 200 response", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 200,
    } as Response);

    const result = await sendSubscriptionToServer(mockSubscription);

    expect(result).toBe(true);
  });

  it("returns true on 204 response", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 204,
    } as Response);

    const result = await sendSubscriptionToServer(mockSubscription);

    expect(result).toBe(true);
  });

  it("returns false on failure", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
    } as Response);

    const result = await sendSubscriptionToServer(mockSubscription);

    expect(result).toBe(false);
  });

  it("returns false on network error", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("Network error"));

    const result = await sendSubscriptionToServer(mockSubscription);

    expect(result).toBe(false);
  });

  it("handles missing keys gracefully", async () => {
    const subWithoutKeys = {
      toJSON: () => ({
        endpoint: "https://push.example.com/sub/123",
        keys: undefined,
      }),
    } as PushSubscription;

    vi.mocked(fetch).mockResolvedValue({ ok: true } as Response);

    await sendSubscriptionToServer(subWithoutKeys);

    const body = JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string);
    expect(body.keys).toEqual({ p256dh: "", auth: "" });
  });
});

describe("removeSubscriptionFromServer", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends endpoint via proxy", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true } as Response);

    await removeSubscriptionFromServer("https://push.example.com/sub/123");

    expect(fetch).toHaveBeenCalledWith("/api/proxy/api/push/unsubscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        endpoint: "https://push.example.com/sub/123",
      }),
    });
  });

  it("returns true on success", async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: true } as Response);

    const result = await removeSubscriptionFromServer(
      "https://push.example.com/sub/123",
    );

    expect(result).toBe(true);
  });

  it("returns true on 204 response", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 204,
    } as Response);

    const result = await removeSubscriptionFromServer(
      "https://push.example.com/sub/123",
    );

    expect(result).toBe(true);
  });

  it("returns false on failure", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
    } as Response);

    const result = await removeSubscriptionFromServer(
      "https://push.example.com/sub/123",
    );

    expect(result).toBe(false);
  });

  it("returns false on network error", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("Network error"));

    const result = await removeSubscriptionFromServer(
      "https://push.example.com/sub/123",
    );

    expect(result).toBe(false);
  });
});
