import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const DATA = { pins: [{ name: "Cafe X", lat: 37.78, lng: -122.4 }] };

const fitBounds = vi.fn();
const addEventListener = vi.fn();
const infoOpen = vi.fn();
const markerCtor = vi.fn();

class FakeMap {
  fitBounds = fitBounds;
}
class FakeBounds {
  extend() {}
}
class FakeInfoWindow {
  close() {}
  setContent() {}
  open = infoOpen;
}
const markers: FakeMarker[] = [];

class FakeMarker {
  map: unknown = null;
  constructor(options: { map: unknown }) {
    markerCtor(options);
    this.map = options.map;
    markers.push(this);
  }
  addEventListener = addEventListener;
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
    vi.useRealTimers();
    fitBounds.mockClear();
    addEventListener.mockClear();
    infoOpen.mockClear();
    markerCtor.mockClear();
    markers.length = 0;
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
    await waitFor(() => expect(markerCtor).toHaveBeenCalledTimes(1));
    expect(screen.getByLabelText("Map")).toBeDefined();
    expect(screen.queryByRole("link", { name: "Cafe X" })).toBeNull();
  });

  it("creates a clickable marker per pin, fits bounds and opens a popup on click", async () => {
    const { MapEmbed: FreshMapEmbed } = await import("../MapEmbed");
    render(
      <FreshMapEmbed
        data={{
          pins: [
            { name: "Cafe X", lat: 37.78, lng: -122.4 },
            { name: "Cafe Y", lat: 37.8, lng: -122.3 },
          ],
        }}
      />,
    );
    await act(async () => {
      await Promise.resolve();
    });
    await waitFor(() => expect(markerCtor).toHaveBeenCalledTimes(2));
    expect(markerCtor.mock.calls[0][0]).toMatchObject({ gmpClickable: true });
    expect(addEventListener).toHaveBeenCalledWith(
      "gmp-click",
      expect.any(Function),
    );
    expect(fitBounds).toHaveBeenCalledTimes(1);
    addEventListener.mock.calls[0][1]();
    expect(infoOpen).toHaveBeenCalledTimes(1);
  });

  it("removes the markers when Google rejects the key after the map is built", async () => {
    const { MapEmbed: FreshMapEmbed } = await import("../MapEmbed");
    render(<FreshMapEmbed data={DATA} />);
    await waitFor(() => expect(markers).toHaveLength(1));
    expect(markers[0].map).not.toBeNull();
    await act(async () => {
      (window as unknown as { gm_authFailure: () => void }).gm_authFailure();
    });
    expect(markers[0].map).toBeNull();
    expect(screen.getByRole("link", { name: "Cafe X" })).toBeDefined();
  });

  it("keeps the pin links when the map loads after the timeout", async () => {
    vi.useFakeTimers();
    const pending: { script?: HTMLScriptElement } = {};
    vi.mocked(document.head.appendChild).mockImplementation(((node: Node) => {
      installFakeMaps();
      pending.script = node as HTMLScriptElement;
      return node;
    }) as typeof document.head.appendChild);
    const { MapEmbed: FreshMapEmbed } = await import("../MapEmbed");
    render(<FreshMapEmbed data={DATA} />);
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });
    expect(screen.getByRole("link", { name: "Cafe X" })).toBeDefined();
    await act(async () => {
      pending.script?.onload?.(new Event("load"));
      for (let i = 0; i < 10; i++) await Promise.resolve();
    });
    expect(markerCtor).not.toHaveBeenCalled();
    expect(screen.getByRole("link", { name: "Cafe X" })).toBeDefined();
  });
});
