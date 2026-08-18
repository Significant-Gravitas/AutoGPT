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
// Module-level mock so tests can override implementations per case.
const downloadArtifactMock = vi.fn(() => Promise.resolve());
vi.mock("../../ArtifactPanel/downloadArtifact", () => ({
  downloadArtifact: (...args: unknown[]) =>
    downloadArtifactMock(...(args as [])),
}));

// Toast is invoked from the download-failure path; spy on it so we
// can assert the user-visible "Download failed" surface fires.
const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...(args as [])),
  useToast: () => ({ toast: toastSpy }),
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
  downloadArtifactMock.mockReset();
  downloadArtifactMock.mockImplementation(() => Promise.resolve());
  toastSpy.mockClear();
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

describe("ArtifactCard — non-openable artifact (download-only)", () => {
  // application/zip + .zip extension is classified as download-only,
  // so the card renders the download-only branch instead of the
  // openable click-to-open variant.
  const ZIP_ARTIFACT: ArtifactRef = {
    id: "660e8400-e29b-41d4-a716-446655440000",
    title: "agent-output.zip",
    mimeType: "application/zip",
    origin: "agent",
    sourceUrl: "https://example.com/file/660e8400-e29b-41d4-a716-446655440000",
  };

  test("clicking the card triggers downloadArtifact", () => {
    render(<ArtifactCard artifact={ZIP_ARTIFACT} />);
    const button = screen.getByRole("button");
    button.click();
    expect(downloadArtifactMock).toHaveBeenCalledTimes(1);
    expect(downloadArtifactMock).toHaveBeenCalledWith(ZIP_ARTIFACT);
    // Non-openable artifacts must NOT open the panel — the panel
    // can't preview them.
    expect(openSpy).not.toHaveBeenCalled();
  });

  test("download failure surfaces a destructive toast", async () => {
    downloadArtifactMock.mockImplementation(() =>
      Promise.reject(new Error("network down")),
    );

    render(<ArtifactCard artifact={ZIP_ARTIFACT} />);
    const button = screen.getByRole("button");
    button.click();

    await vi.waitFor(() => {
      expect(toastSpy).toHaveBeenCalledTimes(1);
    });
    const toastArg = toastSpy.mock.calls[0][0] as {
      title: string;
      variant?: string;
    };
    expect(toastArg.title).toMatch(/download failed/i);
    expect(toastArg.variant).toBe("destructive");
  });
});
