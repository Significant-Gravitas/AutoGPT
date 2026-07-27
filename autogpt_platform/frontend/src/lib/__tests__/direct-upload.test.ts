import { afterEach, describe, expect, it, vi } from "vitest";

import {
  getFileSizeError,
  OAUTH_LOGO_MAX_SIZE_MB,
  SUBMISSION_MEDIA_MAX_SIZE_MB,
  uploadOAuthAppLogoDirect,
  uploadSubmissionMediaDirect,
} from "../direct-upload";

vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn(async () => ({ token: "test-token" })),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "http://backend.test",
  },
}));

interface FakeResponse {
  ok: boolean;
  status: number;
  statusText?: string;
  json?: () => unknown;
}

function mockFetchOnce(response: FakeResponse) {
  const fetchMock = vi.fn(async () => response as unknown as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function makeFile(sizeBytes: number, type = "image/png"): File {
  return new File(["x".repeat(sizeBytes)], "logo.png", { type });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("getFileSizeError", () => {
  it("returns null when the file is within the limit", () => {
    const file = makeFile(1024);
    expect(getFileSizeError(file, SUBMISSION_MEDIA_MAX_SIZE_MB)).toBeNull();
  });

  it("returns a message naming the limit when the file is too large", () => {
    const file = makeFile(OAUTH_LOGO_MAX_SIZE_MB * 1024 * 1024 + 1);
    const message = getFileSizeError(file, OAUTH_LOGO_MAX_SIZE_MB);
    expect(message).toContain("too large");
    expect(message).toContain(`${OAUTH_LOGO_MAX_SIZE_MB}MB`);
  });
});

describe("uploadSubmissionMediaDirect", () => {
  it("bypasses the proxy and hits the backend directly", async () => {
    const fetchMock = mockFetchOnce({
      ok: true,
      status: 200,
      json: () => "https://cdn.test/uploaded.png",
    });

    const url = await uploadSubmissionMediaDirect(makeFile(10));

    expect(url).toBe("https://cdn.test/uploaded.png");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.test/api/store/submissions/media",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("surfaces a clear size message on HTTP 413 instead of a bare status", async () => {
    mockFetchOnce({ ok: false, status: 413, statusText: "Payload Too Large" });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      /too large/i,
    );
  });

  it("surfaces the backend detail message when the backend rejects the size", async () => {
    mockFetchOnce({
      ok: false,
      status: 400,
      json: () => ({ detail: "File too large. Maximum size is 50MB" }),
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "File too large. Maximum size is 50MB",
    );
  });
});

describe("uploadOAuthAppLogoDirect", () => {
  it("posts to the app-scoped logo endpoint", async () => {
    const fetchMock = mockFetchOnce({
      ok: true,
      status: 200,
      json: () => ({}),
    });

    await uploadOAuthAppLogoDirect("app-123", makeFile(10));

    expect(fetchMock).toHaveBeenCalledWith(
      "http://backend.test/api/oauth/apps/app-123/logo/upload",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("throws a clear size message on 413", async () => {
    mockFetchOnce({ ok: false, status: 413, statusText: "Payload Too Large" });

    await expect(
      uploadOAuthAppLogoDirect("app-123", makeFile(10)),
    ).rejects.toThrow(/too large/i);
  });
});
