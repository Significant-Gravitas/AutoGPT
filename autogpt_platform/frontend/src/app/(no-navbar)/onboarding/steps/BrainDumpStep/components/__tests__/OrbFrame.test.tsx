import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_GLASS_PARAMS } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbFrame } from "../OrbFrame";

const { useAudioBarsMock } = vi.hoisted(() => ({
  useAudioBarsMock: vi.fn(() =>
    Array.from({ length: 5 }, () => ({ get: () => 0 })),
  ),
}));

vi.mock("../useAudioBars", () => ({
  useAudioBars: useAudioBarsMock,
}));

vi.mock("../OrbVisual", () => ({
  OrbVisual: function OrbVisual() {
    return <div data-testid="orb-visual" />;
  },
}));

beforeEach(() => {
  useAudioBarsMock.mockClear();
});

describe("OrbFrame", () => {
  it("meters the audio stream only while recording", () => {
    const stream = {} as MediaStream;
    const { rerender } = render(
      <OrbFrame glassParams={DEFAULT_GLASS_PARAMS} audioStream={stream} />,
    );

    expect(useAudioBarsMock).toHaveBeenLastCalledWith(null);

    rerender(
      <OrbFrame
        glassParams={DEFAULT_GLASS_PARAMS}
        audioStream={stream}
        progress={0.5}
      />,
    );

    expect(useAudioBarsMock).toHaveBeenLastCalledWith(stream);
  });
});
