import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import type { ArtifactRef } from "../../../store";
import { ArtifactCard } from "../ArtifactCard";

const registerSpy = vi.fn();
const openSpy = vi.fn();

// Mock the copilot UI store — return values per selector so the
// component can call ``useCopilotUIStore(selector)`` and get back
// our spies / static state.
vi.mock("../../../store", () => {
  function useCopilotUIStore<T>(selector: (state: unknown) => T): T {
    const state = {
      artifactPanel: { isOpen: false, activeArtifact: null },
      openArtifact: openSpy,
      registerArtifactForAutoOpen: registerSpy,
    };
    return selector(state);
  }
  return { useCopilotUIStore };
});

// Skip real download wiring — fetch/blob plumbing isn't under test.
vi.mock("../../ArtifactPanel/downloadArtifact", () => ({
  downloadArtifact: vi.fn(() => Promise.resolve()),
}));

const ARTIFACT: ArtifactRef = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  title: "report.png",
  mimeType: "image/png",
  origin: "agent",
  sourceUrl: "https://example.com/file/550e8400-e29b-41d4-a716-446655440000",
  sizeBytes: 12_345,
};

beforeEach(() => {
  registerSpy.mockClear();
  openSpy.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("ArtifactCard — readOnly", () => {
  test("does NOT register for auto-open in readOnly mode", () => {
    render(<ArtifactCard artifact={ARTIFACT} readOnly />);
    expect(registerSpy).not.toHaveBeenCalled();
  });

  test("registers for auto-open in default (owner) mode", () => {
    render(<ArtifactCard artifact={ARTIFACT} />);
    expect(registerSpy).toHaveBeenCalledTimes(1);
    expect(registerSpy).toHaveBeenCalledWith(ARTIFACT);
  });

  test("openable artifact opens the panel on click in readOnly mode", () => {
    // Click on an openable artifact still calls openArtifact even in readOnly —
    // that's how the share viewer renders previews via ArtifactPanel.
    render(<ArtifactCard artifact={ARTIFACT} readOnly />);
    const button = screen.getByRole("button");
    button.click();
    expect(openSpy).toHaveBeenCalledWith(ARTIFACT);
  });

  test("renders the artifact title", () => {
    render(<ArtifactCard artifact={ARTIFACT} readOnly />);
    expect(screen.getByText("report.png")).toBeDefined();
  });
});
