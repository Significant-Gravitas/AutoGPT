import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { AuthTokenUnavailableError } from "@/lib/autogpt-server-api/helpers";
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  GET as getChatStream,
  POST as postChatStream,
} from "../chat/sessions/[sessionId]/stream/route";
import { POST as postTranscription } from "../transcribe/route";
import { POST as postWorkspaceUpload } from "../workspace/files/upload/route";

vi.mock("@/lib/auth/server/getServerAuthToken", () => ({
  getServerAuthToken: vi.fn(),
}));

function formRequest(url: string, field: string, type: string) {
  const formData = new FormData();
  formData.append(field, new Blob(["bytes"], { type }));
  return new NextRequest(url, { method: "POST", body: formData });
}

function sessionParams() {
  return { params: Promise.resolve({ sessionId: "session-1" }) };
}

const routes = [
  {
    name: "transcribe",
    call: () =>
      postTranscription(
        formRequest("https://app.test/api/transcribe", "audio", "audio/webm"),
      ),
  },
  {
    name: "workspace file upload",
    call: () =>
      postWorkspaceUpload(
        formRequest(
          "https://app.test/api/workspace/files/upload?session_id=session-1",
          "file",
          "text/plain",
        ),
      ),
  },
  {
    name: "chat stream POST",
    call: () =>
      postChatStream(
        new NextRequest("https://app.test/api/chat/sessions/session-1/stream", {
          method: "POST",
          body: JSON.stringify({ message: "hello" }),
        }),
        sessionParams(),
      ),
  },
  {
    name: "chat stream GET",
    call: () =>
      getChatStream(
        new NextRequest("https://app.test/api/chat/sessions/session-1/stream"),
        sessionParams(),
      ),
  },
];

describe("route handlers that cannot mint an auth token", () => {
  beforeEach(() => {
    vi.mocked(getServerAuthToken).mockRejectedValue(
      new AuthTokenUnavailableError(),
    );
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(getServerAuthToken).mockReset();
  });

  it.each(routes)(
    "$name answers a retryable 503 without calling the backend",
    async ({ call }) => {
      const response = await call();

      expect(fetch).not.toHaveBeenCalled();
      expect(response.status).toBe(503);
      expect(response.headers.get("retry-after")).toBe("2");
      expect(await response.json()).toMatchObject({
        code: "auth_token_unavailable",
      });
    },
  );
});
