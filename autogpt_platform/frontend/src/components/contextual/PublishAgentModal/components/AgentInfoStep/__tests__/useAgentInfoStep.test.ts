import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

import { useAgentInfoStep } from "../useAgentInfoStep";

describe("useAgentInfoStep", () => {
  const baseArgs = {
    onBack: vi.fn(),
    onSuccess: vi.fn(),
    selectedAgentId: "graph-1",
    selectedAgentVersion: 1,
  };

  it("clears the image-required error once an image is added so submit re-enables", () => {
    const { result } = renderHook(() => useAgentInfoStep(baseArgs));

    // Simulate submitting without a thumbnail: the form marks the root error.
    act(() => {
      result.current.form.setError("root", {
        type: "manual",
        message: "At least one image is required",
      });
    });
    expect(result.current.form.formState.errors.root).toBeDefined();

    // Adding an image must clear the stale error, otherwise the submit button
    // stays disabled forever and the user is stuck.
    act(() => {
      result.current.handleImagesChange(["https://example.com/thumb.png"]);
    });

    expect(result.current.form.formState.errors.root).toBeUndefined();
    expect(result.current.images).toEqual(["https://example.com/thumb.png"]);
  });

  it("keeps the error while there are still no images", () => {
    const { result } = renderHook(() => useAgentInfoStep(baseArgs));

    act(() => {
      result.current.form.setError("root", {
        type: "manual",
        message: "At least one image is required",
      });
    });

    act(() => {
      result.current.handleImagesChange([]);
    });

    expect(result.current.form.formState.errors.root).toBeDefined();
  });
});
