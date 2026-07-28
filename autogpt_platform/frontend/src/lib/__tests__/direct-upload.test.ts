import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getFileSizeError,
  OAUTH_LOGO_MAX_SIZE_MB,
  SUBMISSION_MEDIA_MAX_SIZE_MB,
  uploadFileDirect,
  uploadOAuthAppLogoDirect,
  uploadSubmissionMediaDirect,
} from "../direct-upload";

const getTokenMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: getTokenMock,
}));

beforeEach(() => {
  getTokenMock.mockResolvedValue({ token: "test-token" });
});

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
  text?: () => string | Promise<string>;
}

function mockFetchOnce(response: FakeResponse) {
  const fetchMock = vi.fn(
    async (_url: string, _init?: RequestInit) =>
      response as unknown as Response,
  );
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
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer test-token" },
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.body).toBeInstanceOf(FormData);
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
      text: () =>
        JSON.stringify({ detail: "File too large. Maximum size is 50MB" }),
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
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer test-token" },
      }),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.body).toBeInstanceOf(FormData);
  });

  it("throws a clear size message on 413", async () => {
    mockFetchOnce({ ok: false, status: 413, statusText: "Payload Too Large" });

    await expect(
      uploadOAuthAppLogoDirect("app-123", makeFile(10)),
    ).rejects.toThrow(/too large/i);
  });
});

describe("uploadFileDirect (workspace)", () => {
  it("hits the workspace endpoint with overwrite and session params", async () => {
    const fetchMock = mockFetchOnce({
      ok: true,
      status: 200,
      json: () => ({
        file_id: "f1",
        name: "a.png",
        path: "p",
        mime_type: "image/png",
        size_bytes: 10,
      }),
    });

    const result = await uploadFileDirect(makeFile(10), "sess-1");

    expect(result.file_id).toBe("f1");
    const calledUrl = fetchMock.mock.calls[0]?.[0] as string;
    expect(calledUrl).toContain(
      "http://backend.test/api/workspace/files/upload",
    );
    expect(calledUrl).toContain("overwrite=true");
    expect(calledUrl).toContain("session_id=sess-1");
  });
});

describe("authentication", () => {
  it("throws an auth error when no token is available", async () => {
    getTokenMock.mockResolvedValueOnce({ token: null, error: "no session" });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      /sign in again/i,
    );
  });
});

describe("readUploadError fallbacks", () => {
  it("surfaces the raw response body when it is not JSON", async () => {
    mockFetchOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: () => "upstream connection reset",
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "upstream connection reset",
    );
  });

  it("falls back to statusText when the body is empty", async () => {
    mockFetchOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: () => "",
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "Internal Server Error",
    );
  });

  it("falls back to the HTTP status when there is no statusText or body", async () => {
    mockFetchOnce({
      ok: false,
      status: 500,
      statusText: "",
      text: () => "",
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "Upload failed (HTTP 500)",
    );
  });

  it("reads a nested detail.message object", async () => {
    mockFetchOnce({
      ok: false,
      status: 400,
      text: () =>
        JSON.stringify({ detail: { message: "nested detail message" } }),
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "nested detail message",
    );
  });

  it("reads a top-level message field", async () => {
    mockFetchOnce({
      ok: false,
      status: 400,
      text: () => JSON.stringify({ message: "top level message" }),
    });

    await expect(uploadSubmissionMediaDirect(makeFile(10))).rejects.toThrow(
      "top level message",
    );
  });
});
