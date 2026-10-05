import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { TERMS_VERSION } from "@/lib/legal";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  postV1RecordUserConsent: vi.fn(),
  captureException: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1RecordUserConsent: mocks.postV1RecordUserConsent,
}));
vi.mock("@sentry/nextjs", () => ({
  captureException: mocks.captureException,
}));

import { recordSignupConsent } from "../recordSignupConsent";

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  mocks.postV1RecordUserConsent.mockResolvedValue({ status: 200, data: {} });
});

describe("recordSignupConsent", () => {
  it.each([true, false])(
    "records the current terms version with marketing_opt_out=%s",
    async (marketingOptOut) => {
      await recordSignupConsent({ userID: "user-1", marketingOptOut });

      expect(mocks.postV1RecordUserConsent).toHaveBeenCalledOnce();
      expect(mocks.postV1RecordUserConsent).toHaveBeenCalledWith({
        terms_version: TERMS_VERSION,
        marketing_opt_out: marketingOptOut,
      });
      expect(mocks.captureException).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["an ApiError from a non-2xx response", new ApiError("Bad", 500, {})],
    ["a network failure", new TypeError("Failed to fetch")],
  ])("reports %s without throwing", async (_label, error) => {
    mocks.postV1RecordUserConsent.mockRejectedValue(error);

    await expect(
      recordSignupConsent({ userID: "user-1", marketingOptOut: true }),
    ).resolves.toBeUndefined();

    expect(mocks.captureException).toHaveBeenCalledOnce();
    expect(mocks.captureException).toHaveBeenCalledWith(error, {
      tags: { signup_step: "record_consent" },
      user: { id: "user-1" },
      extra: { marketingOptOut: true },
    });
    expect(console.error).toHaveBeenCalledWith(
      "Failed to record signup consent:",
      error,
    );
  });
});
