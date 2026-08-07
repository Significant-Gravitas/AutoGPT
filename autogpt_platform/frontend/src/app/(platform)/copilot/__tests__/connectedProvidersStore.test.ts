import { renderHook, act } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  useAreAllConnected,
  useConnectedProvidersStore,
} from "../connectedProvidersStore";

afterEach(() => {
  // Reset the store so tests are isolated.
  useConnectedProvidersStore.setState({
    connected: new Set(),
    autoDismissedKeys: new Set(),
  });
});

describe("connectedProvidersStore", () => {
  it("marks providers connected within a session", () => {
    const { result } = renderHook(() =>
      useAreAllConnected("sess-1", ["github"]),
    );
    expect(result.current).toBe(false);

    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["github"] });
    });
    expect(result.current).toBe(true);
  });

  it("scopes connections per session", () => {
    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["github"] });
    });

    const sameSession = renderHook(() =>
      useAreAllConnected("sess-1", ["github"]),
    );
    const otherSession = renderHook(() =>
      useAreAllConnected("sess-2", ["github"]),
    );
    expect(sameSession.result.current).toBe(true);
    expect(otherSession.result.current).toBe(false);
  });

  it("returns false until ALL required providers are connected", () => {
    const { result } = renderHook(() =>
      useAreAllConnected("sess-1", ["github", "openai"]),
    );

    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["github"] });
    });
    expect(result.current).toBe(false);

    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["openai"] });
    });
    expect(result.current).toBe(true);
  });

  it("returns false when sessionID is null", () => {
    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["github"] });
    });

    const { result } = renderHook(() => useAreAllConnected(null, ["github"]));
    expect(result.current).toBe(false);
  });

  it("returns false when no providers are requested", () => {
    const { result } = renderHook(() => useAreAllConnected("sess-1", []));
    expect(result.current).toBe(false);
  });

  it("clearSession removes only that session's marks", () => {
    act(() => {
      const store = useConnectedProvidersStore.getState();
      store.markConnected({ sessionID: "sess-1", providers: ["github"] });
      store.markConnected({ sessionID: "sess-2", providers: ["openai"] });
    });

    act(() => {
      useConnectedProvidersStore.getState().clearSession("sess-1");
    });

    const s1 = renderHook(() => useAreAllConnected("sess-1", ["github"]));
    const s2 = renderHook(() => useAreAllConnected("sess-2", ["openai"]));
    expect(s1.result.current).toBe(false);
    expect(s2.result.current).toBe(true);
  });

  it("ignores empty/falsy providers in markConnected", () => {
    act(() => {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID: "sess-1", providers: ["", "github"] });
    });
    expect(useConnectedProvidersStore.getState().connected.size).toBe(1);
  });

  describe("tryClaimAutoDismiss", () => {
    it("returns true on first claim and false on subsequent claims", () => {
      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        }),
      ).toBe(false);
    });

    it("treats different provider sets in the same session as separate slots", () => {
      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["openai"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github", "openai"],
        }),
      ).toBe(true);
    });

    it("treats the same provider set in different sessions as separate slots", () => {
      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-2",
          providers: ["github"],
        }),
      ).toBe(true);
    });

    it("ignores provider order when matching slots", () => {
      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github", "openai"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["openai", "github"],
        }),
      ).toBe(false);
    });

    it("returns false when sessionID is empty or providers list is empty", () => {
      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({ sessionID: "", providers: ["github"] }),
      ).toBe(false);
      expect(
        store.tryClaimAutoDismiss({ sessionID: "sess-1", providers: [] }),
      ).toBe(false);
    });

    it("clearSession releases auto-dismiss slots scoped to that session", () => {
      act(() => {
        const store = useConnectedProvidersStore.getState();
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        });
        store.tryClaimAutoDismiss({
          sessionID: "sess-2",
          providers: ["github"],
        });
      });

      act(() => {
        useConnectedProvidersStore.getState().clearSession("sess-1");
      });

      const store = useConnectedProvidersStore.getState();
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-1",
          providers: ["github"],
        }),
      ).toBe(true);
      expect(
        store.tryClaimAutoDismiss({
          sessionID: "sess-2",
          providers: ["github"],
        }),
      ).toBe(false);
    });
  });
});
