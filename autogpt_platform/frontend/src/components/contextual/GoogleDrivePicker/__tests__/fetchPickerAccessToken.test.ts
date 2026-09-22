import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  postV1GetPickerToken: vi.fn(),
  getGetV1GetSpecificCredentialByIdQueryOptions: vi.fn(),
}));

import { postV1GetPickerToken } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { fetchPickerAccessToken } from "../useGoogleDrivePicker";

// Keep the cast rather than `vi.mocked(postV1GetPickerToken)` so the
// intentionally-malformed mock responses below (missing `headers`, empty
// `data`, non-200 status) don't have to satisfy the generated
// `postV1GetPickerTokenResponse` union — these tests exercise defensive
// paths against misshapen server responses.
const mockPost = postV1GetPickerToken as unknown as ReturnType<typeof vi.fn>;

afterEach(() => {
  mockPost.mockReset();
});

describe("fetchPickerAccessToken", () => {
  it("returns the access token when the picker-token endpoint succeeds", async () => {
    // Backend `POST /api/integrations/google/credentials/{id}/picker-token`
    // returns { access_token, access_token_expires_at } on success. The hook
    // only needs the access_token; `access_token_expires_at` is informational.
    mockPost.mockResolvedValue({
      status: 200,
      data: {
        access_token: "ya29.real-token-value",
        access_token_expires_at: 1776834363,
      },
    });

    await expect(fetchPickerAccessToken("cred-123")).resolves.toBe(
      "ya29.real-token-value",
    );
    expect(mockPost).toHaveBeenCalledWith("google", "cred-123");
  });

  it("throws when the server returns a non-200 response", async () => {
    // Backend returns 400/404 for non-OAuth2 creds, missing creds, or creds
    // without an access_token. `okData` treats anything !== 200 as undefined,
    // so we should surface a distinct error rather than silently passing an
    // empty string to the Google Picker.
    mockPost.mockResolvedValue({
      status: 400,
      data: {
        detail: "Picker tokens are only available for OAuth2 credentials",
      },
    });

    await expect(fetchPickerAccessToken("cred-456")).rejects.toThrow(
      /did not return an access token/i,
    );
  });

  it("throws when the 200 response is missing access_token", async () => {
    // Regression guard for the pre-PR-#12874 shape: the meta-only endpoint
    // used to satisfy the picker query but never included access_token —
    // the hook silently entered the "Failed to retrieve" fallback path.
    // If the server ever responds with an empty `data` again, we want a
    // loud throw, not a silent fallback, so reviewers see the mismatch.
    mockPost.mockResolvedValue({ status: 200, data: {} });

    await expect(fetchPickerAccessToken("cred-789")).rejects.toThrow(
      /did not return an access token/i,
    );
  });

  it("throws when the 200 response has an empty-string access_token", async () => {
    // Defence against a misconfigured or stripped server response. An empty
    // string would pass `response.data.access_token` presence checks in some
    // implementations; make sure falsy-but-present also trips the guard.
    mockPost.mockResolvedValue({
      status: 200,
      data: { access_token: "", access_token_expires_at: null },
    });

    await expect(fetchPickerAccessToken("cred-empty")).rejects.toThrow(
      /did not return an access token/i,
    );
  });

  it("propagates thrown errors from the underlying fetch", async () => {
    // Network failure / proxy 415 / JSON parse error surfaces as a thrown
    // error from the generated mutator. Don't swallow it — the caller
    // (`useGoogleDrivePicker.openPicker()`) has its own try/catch that
    // surfaces an "Authentication Error" toast, which relies on this
    // propagation.
    mockPost.mockRejectedValue(new Error("network boom"));

    await expect(fetchPickerAccessToken("cred-boom")).rejects.toThrow(
      "network boom",
    );
  });
});
