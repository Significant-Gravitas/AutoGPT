import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  getGraphLoadErrorToast,
  retryUnlessClientError,
} from "../graphLoadError";

describe("getGraphLoadErrorToast", () => {
  it("labels a 404 as not found and surfaces the backend message", () => {
    const error = new ApiError("Graph not found", 404, { detail: "..." });
    const toast = getGraphLoadErrorToast(error);
    expect(toast.title).toBe("Agent not found");
    expect(toast.description).toBe("Graph not found");
  });

  it("labels a 401/403 as unauthorized and surfaces the backend message", () => {
    for (const status of [401, 403]) {
      const error = new ApiError("Not authorized", status, {});
      const toast = getGraphLoadErrorToast(error);
      expect(toast.title).toBe("Not authorized to view this agent");
      expect(toast.description).toBe("Not authorized");
    }
  });

  it("falls back to a generic failure title for other statuses", () => {
    const error = new ApiError("Internal server error", 500, {});
    const toast = getGraphLoadErrorToast(error);
    expect(toast.title).toBe("Failed to load agent");
    expect(toast.description).toBe("Internal server error");
  });

  it("falls back to a generic description for a non-ApiError", () => {
    const toast = getGraphLoadErrorToast(new Error());
    expect(toast.title).toBe("Failed to load agent");
    expect(toast.description).toBe("An unexpected error occurred.");
  });

  it("falls back to a generic description for a non-Error value", () => {
    const toast = getGraphLoadErrorToast("boom");
    expect(toast.title).toBe("Failed to load agent");
    expect(toast.description).toBe("An unexpected error occurred.");
  });
});

describe("retryUnlessClientError", () => {
  it("does not retry a 404", () => {
    expect(retryUnlessClientError(0, new ApiError("nope", 404, {}))).toBe(
      false,
    );
  });

  it("does not retry a 401", () => {
    expect(retryUnlessClientError(0, new ApiError("nope", 401, {}))).toBe(
      false,
    );
  });

  it("retries a 500 up to 3 attempts", () => {
    const error = new ApiError("boom", 500, {});
    expect(retryUnlessClientError(0, error)).toBe(true);
    expect(retryUnlessClientError(2, error)).toBe(true);
    expect(retryUnlessClientError(3, error)).toBe(false);
  });

  it("retries a non-ApiError up to 3 attempts", () => {
    expect(retryUnlessClientError(0, new Error("boom"))).toBe(true);
    expect(retryUnlessClientError(3, new Error("boom"))).toBe(false);
  });
});
