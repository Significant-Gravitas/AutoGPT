import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const DATA = { pins: [{ name: "Cafe X", lat: 37.78, lng: -122.4 }] };

class FakeMap {}
class FakeBounds {
  extend() {}
}
class FakeInfoWindow {}
class FakeMarker {
  addListener() {}
}

function installFakeMaps() {
  (window as unknown as { google: unknown }).google = {
    maps: {
      Map: FakeMap,
      LatLngBounds: FakeBounds,
      InfoWindow: FakeInfoWindow,
      marker: { AdvancedMarkerElement: FakeMarker },
      importLibrary: () => Promise.resolve({}),
    },
  };
}

describe("MapEmbed with a Maps key", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY", "test-key");
    vi.spyOn(document.head, "appendChild").mockImplementation(((node: Node) => {
      installFakeMaps();
      queueMicrotask(() =>
        (node as HTMLScriptElement).onload?.(new Event("load")),
      );
      return node;
    }) as typeof document.head.appendChild);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("falls back to pin links when Google rejects the key", async () => {
    const { MapEmbed: FreshMapEmbed } = await import("../MapEmbed");
    render(<FreshMapEmbed data={DATA} />);
    expect(screen.queryByRole("link", { name: "Cafe X" })).toBeNull();
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      (window as unknown as { gm_authFailure: () => void }).gm_authFailure();
    });
    await waitFor(() =>
      expect(screen.getByRole("link", { name: "Cafe X" })).toBeDefined(),
    );
  });

  it("keeps the map container when loading succeeds", async () => {
    const { MapEmbed: FreshMapEmbed } = await import("../MapEmbed");
    render(<FreshMapEmbed data={DATA} />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByLabelText("Map")).toBeDefined();
    expect(screen.queryByRole("link", { name: "Cafe X" })).toBeNull();
  });
});
