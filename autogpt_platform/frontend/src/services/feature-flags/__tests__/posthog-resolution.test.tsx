import { PostHogProvider } from "@posthog/react";
import {
  act,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { PostHog, type PostHogConfig } from "posthog-js";
import { type ReactNode, useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.hoisted(() => {
  process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = "posthog";
  process.env.NEXT_PUBLIC_BEHAVE_AS = "CLOUD";
  process.env.NEXT_PUBLIC_POSTHOG_KEY = "phc_test";
  process.env.NEXT_PUBLIC_POSTHOG_HOST = "https://posthog.test";
});

import { trackPostHogFlagResolution } from "../backends/posthog-resolution";
import { Flag, useFlagStatus, useGetFlag } from "../use-get-flag";
import { MemoryPageAccess } from "@/app/(platform)/settings/memory/components/MemoryPageAccess";

const router = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
}));

const cleanups: Array<() => void> = [];
let clientNumber = 0;

beforeEach(() => router.push.mockClear());

afterEach(() => {
  cleanups.splice(0).forEach((cleanup) => cleanup());
  vi.useRealTimers();
});

function createClient(bootstrap?: PostHogConfig["bootstrap"]) {
  const client = new PostHog();
  cleanups.push(trackPostHogFlagResolution(client));
  client.init("phc_flag_defaults_test", {
    api_host: "https://posthog.test",
    persistence: "memory",
    persistence_name: `flag-defaults-${clientNumber++}`,
    advanced_disable_flags: true,
    disable_external_dependency_loading: true,
    disable_session_recording: true,
    opt_out_capturing_by_default: true,
    capture_pageview: false,
    capture_pageleave: false,
    autocapture: false,
    bootstrap,
  });
  return client;
}

function withClient(client: PostHog) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <PostHogProvider client={client}>{children}</PostHogProvider>;
  };
}

function receive(client: PostHog, flags: Record<string, boolean>) {
  client.featureFlags.receivedFeatureFlags(
    {
      errorsWhileComputingFlags: false,
      flags: Object.fromEntries(
        Object.entries(flags).map(([key, enabled]) => [
          key,
          {
            key,
            enabled,
            variant: undefined,
            reason: undefined,
            metadata: undefined,
          },
        ]),
      ),
    },
    false,
  );
}

describe("real PostHog SDK resolution", () => {
  it("uses the mature fallback before the vendor answers", () => {
    const client = createClient();

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });

    expect(result.current).toEqual({
      enabled: true,
      ready: false,
      answered: false,
    });
  });

  it("does not call an earlier failed request an answer for a late-mounted hook", () => {
    const client = createClient();
    client.featureFlags.receivedFeatureFlags({}, true);
    expect(client.featureFlags.hasLoadedFlags).toBe(true);

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });

    expect(result.current).toEqual({
      enabled: true,
      ready: false,
      answered: false,
    });
  });

  it("resolves an omitted inactive flag after a complete response", () => {
    const client = createClient();
    receive(client, {});

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });
    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("serves an earlier explicit off to a late-mounted hook as answered", () => {
    const client = createClient();
    receive(client, { [Flag.HIRE_EXPERTS]: false });

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });

    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("preserves the default and readiness of an omitted unselected flag", () => {
    const client = createClient();
    receive(client, {});

    const { result } = renderHook(
      () => useFlagStatus(Flag.ENABLE_PLATFORM_PAYMENT),
      { wrapper: withClient(client) },
    );

    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("keeps a bootstrapped value without calling it a live answer", () => {
    const client = createClient({
      featureFlags: { [Flag.ENABLE_PLATFORM_PAYMENT]: true },
    });

    const { result } = renderHook(
      () => useFlagStatus(Flag.ENABLE_PLATFORM_PAYMENT),
      { wrapper: withClient(client) },
    );

    expect(result.current).toEqual({
      enabled: true,
      ready: false,
      answered: false,
    });
  });

  it("retains cached values without authority after a reload fails", () => {
    const client = createClient();
    receive(client, { [Flag.HIRE_EXPERTS]: true });
    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });
    expect(result.current.answered).toBe(true);

    act(() => client.featureFlags.receivedFeatureFlags({}, true));

    expect(result.current.enabled).toBe(true);
    expect(result.current.answered).toBe(false);
  });

  it("updates the value when another successful answer keeps the same authority", () => {
    const client = createClient();
    receive(client, { [Flag.HIRE_EXPERTS]: true });
    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });
    expect(result.current).toMatchObject({ enabled: true, answered: true });

    act(() => receive(client, { [Flag.HIRE_EXPERTS]: false }));

    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("commits the refreshed value and its authority together", async () => {
    const client = createClient();
    receive(client, { [Flag.HIRE_EXPERTS]: false });
    client.featureFlags.receivedFeatureFlags({}, true);
    const committed: Array<{ enabled: boolean; answered: boolean }> = [];
    const { result } = renderHook(
      () => {
        const status = useFlagStatus(Flag.HIRE_EXPERTS);
        useEffect(() => {
          committed.push({
            enabled: status.enabled,
            answered: status.answered,
          });
        }, [status.enabled, status.answered]);
        return status;
      },
      { wrapper: withClient(client) },
    );
    expect(result.current).toMatchObject({ enabled: false, answered: false });

    await new Promise<void>((resolve) => {
      setTimeout(() => {
        receive(client, { [Flag.HIRE_EXPERTS]: true });
        setTimeout(resolve, 30);
      }, 0);
    });
    await waitFor(() =>
      expect(result.current).toMatchObject({ enabled: true, answered: true }),
    );

    expect(committed).not.toContainEqual({ enabled: false, answered: true });
  });

  it("does not call a flag omitted by a partial evaluation authoritative", () => {
    const client = createClient();
    client.featureFlags.receivedFeatureFlags(
      { flags: {}, errorsWhileComputingFlags: true },
      false,
    );

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withClient(client),
    });

    expect(result.current.answered).toBe(false);
  });

  it.each([false, true])(
    "keeps cached %s without authority when a partial response omits it",
    (cached) => {
      const client = createClient();
      receive(client, { [Flag.GRAPHITI_MEMORY]: cached });
      const { result } = renderHook(() => useFlagStatus(Flag.GRAPHITI_MEMORY), {
        wrapper: withClient(client),
      });
      expect(result.current.answered).toBe(true);

      act(() =>
        client.featureFlags.receivedFeatureFlags(
          { flags: {}, errorsWhileComputingFlags: true },
          false,
        ),
      );

      expect(result.current).toMatchObject({
        enabled: cached,
        answered: false,
      });

      act(() =>
        client.featureFlags.receivedFeatureFlags(
          {
            featureFlags: { [Flag.GRAPHITI_MEMORY]: false },
            errorsWhileComputingFlags: true,
          },
          false,
        ),
      );
      expect(result.current).toMatchObject({ enabled: false, answered: true });
    },
  );

  it.each([
    { flags: {}, errorsWhileComputingFlags: false },
    { featureFlags: {}, errorsWhileComputingFlags: false },
  ])(
    "finishes the memory route guard after a complete omission: %j",
    (response) => {
      const client = createClient();
      render(
        <MemoryPageAccess>
          <div>Memory contents</div>
        </MemoryPageAccess>,
        { wrapper: withClient(client) },
      );
      expect(screen.getByRole("status")).toBeTruthy();

      act(() => client.featureFlags.receivedFeatureFlags(response, false));

      expect(screen.queryByRole("status")).toBeNull();
      expect(screen.queryByText("Memory contents")).toBeNull();
      expect(router.push).toHaveBeenCalledWith("/404");
    },
  );

  it.each([
    { response: { flags: {}, errorsWhileComputingFlags: true }, failed: false },
    { response: { flags: {} }, failed: false },
    { response: { errorsWhileComputingFlags: false }, failed: false },
    { response: { flags: {}, errorsWhileComputingFlags: false }, failed: true },
  ])(
    "does not deny memory access on an uncertain omission: %j",
    ({ response, failed }) => {
      vi.useFakeTimers();
      const client = createClient();
      render(
        <MemoryPageAccess>
          <div>Memory contents</div>
        </MemoryPageAccess>,
        { wrapper: withClient(client) },
      );

      act(() => client.featureFlags.receivedFeatureFlags(response, failed));
      act(() => vi.advanceTimersByTime(10_000));

      expect(screen.getByRole("status")).toBeTruthy();
      expect(router.push).not.toHaveBeenCalled();
      expect(screen.queryByText("Memory contents")).toBeNull();

      act(() => receive(client, { [Flag.GRAPHITI_MEMORY]: true }));

      expect(screen.queryByRole("status")).toBeNull();
      expect(screen.getByText("Memory contents")).toBeTruthy();
      expect(router.push).not.toHaveBeenCalled();
    },
  );

  it("restores the SDK receiver after the last observer and distrusts untracked responses", () => {
    const original = new PostHog().featureFlags.receivedFeatureFlags;
    const client = createClient();
    const tracked = client.featureFlags.receivedFeatureFlags;
    const stopSecond = trackPostHogFlagResolution(client);
    receive(client, {});

    cleanups.pop()?.();
    expect(client.featureFlags.receivedFeatureFlags).toBe(tracked);
    stopSecond();
    expect(client.featureFlags.receivedFeatureFlags).toBe(original);

    client.featureFlags.receivedFeatureFlags(
      { flags: {}, errorsWhileComputingFlags: true },
      false,
    );
    cleanups.push(trackPostHogFlagResolution(client));
    const { result } = renderHook(() => useFlagStatus(Flag.GRAPHITI_MEMORY), {
      wrapper: withClient(client),
    });
    expect(result.current.answered).toBe(false);

    act(() => receive(client, {}));
    expect(result.current).toEqual({
      enabled: false,
      ready: true,
      answered: true,
    });
  });

  it("does not carry live authority into a contextless update after reset", () => {
    const client = createClient();
    receive(client, { [Flag.GRAPHITI_MEMORY]: true });
    const { result } = renderHook(() => useFlagStatus(Flag.GRAPHITI_MEMORY), {
      wrapper: withClient(client),
    });
    expect(result.current.answered).toBe(true);

    act(() => {
      client.reset();
      client.featureFlags.receivedFeatureFlags({
        featureFlags: { [Flag.GRAPHITI_MEMORY]: true },
        errorsWhileComputingFlags: false,
      });
    });

    expect(result.current).toMatchObject({ enabled: true, answered: false });
  });

  it("preserves a later SDK receiver replacement during cleanup", () => {
    const client = createClient();
    const replacement = vi.fn();
    client.featureFlags.receivedFeatureFlags = replacement;

    cleanups.pop()?.();

    expect(client.featureFlags.receivedFeatureFlags).toBe(replacement);
  });

  it("uses the vendor's search terms payload", () => {
    const client = createClient();
    client.featureFlags.receivedFeatureFlags(
      {
        flags: {
          [Flag.MARKETPLACE_SEARCH_TERMS]: {
            key: Flag.MARKETPLACE_SEARCH_TERMS,
            enabled: true,
            variant: undefined,
            reason: undefined,
            metadata: {
              id: 1,
              version: 1,
              description: undefined,
              payload: JSON.stringify(["Design", "Support"]),
            },
          },
        },
      },
      false,
    );

    const { result } = renderHook(
      () => useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS),
      { wrapper: withClient(client) },
    );

    expect(result.current).toEqual(["Design", "Support"]);
  });
});
