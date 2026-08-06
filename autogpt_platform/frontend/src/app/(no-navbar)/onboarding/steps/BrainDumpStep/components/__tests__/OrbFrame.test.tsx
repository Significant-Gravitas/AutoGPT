import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_GLASS_PARAMS } from "@/components/molecules/GlassOrb/GlassSurface";
import { OrbFrame } from "../OrbFrame";
import { DEFAULT_WAVY_ORB_SETTINGS } from "../WavyOrb/helpers";

const { useAudioLevelMock } = vi.hoisted(() => ({
  useAudioLevelMock: vi.fn(() => ({ get: () => 0 })),
}));

vi.mock("../useAudioLevel", () => ({
  useAudioLevel: useAudioLevelMock,
}));

vi.mock("../OrbVisual", () => ({
  OrbVisual: function OrbVisual() {
    return <div data-testid="orb-visual" />;
  },
}));

beforeEach(() => {
  useAudioLevelMock.mockClear();
});

describe("OrbFrame", () => {
  it("does not start the shared audio analyzer for the wavy variant", () => {
    const stream = {} as MediaStream;
    const { rerender } = render(
      <OrbFrame
        glassParams={DEFAULT_GLASS_PARAMS}
        variant="wavy"
        audioStream={stream}
        wavySettings={DEFAULT_WAVY_ORB_SETTINGS}
        progress={0.5}
      />,
    );

    expect(useAudioLevelMock).toHaveBeenLastCalledWith(null);

    rerender(
      <OrbFrame
        glassParams={DEFAULT_GLASS_PARAMS}
        variant="glass"
        audioStream={stream}
        wavySettings={DEFAULT_WAVY_ORB_SETTINGS}
        progress={0.5}
      />,
    );

    expect(useAudioLevelMock).toHaveBeenLastCalledWith(stream);
  });
});
