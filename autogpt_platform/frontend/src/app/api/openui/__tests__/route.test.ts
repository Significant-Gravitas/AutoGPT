import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/server/getServerAuthToken", () => ({
  getServerAuthToken: vi.fn(),
}));

import { getServerAuthToken } from "@/lib/auth/server/getServerAuthToken";
import { POST } from "../route";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

function request(body: unknown) {
  return new Request("http://localhost/api/openui", {
    method: "POST",
    headers: { "content-type": "application/json", origin: "http://localhost" },
    body: JSON.stringify(body),
  });
}

describe("OpenUI generation boundary", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_OPENUI_EXPERIMENT", "true");
    vi.stubEnv("OPENUI_API_KEY", "");
    vi.mocked(getServerAuthToken).mockResolvedValue("test-session");
  });
  afterEach(() => vi.unstubAllEnvs());

  it("rejects requests from another origin", async () => {
    const input = request({ prompt: "Hello" });
    input.headers.set("origin", "https://another.example");
    expect((await POST(input)).status).toBe(403);
  });

  it("streams a provider response and marks its completion", async () => {
    vi.stubEnv("OPENUI_API_KEY", "test-provider-key");
    vi.stubEnv("OPENUI_BASE_URL", "https://model.example/v1");
    vi.stubEnv("OPENUI_MODEL", "test-model");
    const source = 'root = Workspace("Live result", "From the model", [])';
    server.use(
      http.post(
        "https://model.example/v1/chat/completions",
        async ({ request }) => {
          const body = (await request.json()) as {
            messages: { role: string; content: string }[];
          };
          expect(body.messages[0].content).toContain("Workspace");
          expect(body.messages[1].content).toContain("Show a dashboard");
          return new HttpResponse(
            [
              `data: ${JSON.stringify({ id: "test", choices: [{ index: 0, delta: { content: source }, finish_reason: null }] })}\n\n`,
              `data: ${JSON.stringify({ id: "test", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`,
              "data: [DONE]\n\n",
            ].join(""),
            { headers: { "Content-Type": "text/event-stream" } },
          );
        },
      ),
    );
    const response = await POST(request({ prompt: "Show a dashboard" }));
    const events = (await response.text())
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(events).toEqual([{ type: "delta", text: source }, { type: "done" }]);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
  });

  it("rejects a bounded request body even when Content-Length is absent", async () => {
    expect(
      (await POST(request({ prompt: "Hello", ignored: "x".repeat(81_000) })))
        .status,
    ).toBe(400);
  });

  it("is unavailable unless explicitly enabled", async () => {
    vi.stubEnv("NEXT_PUBLIC_OPENUI_EXPERIMENT", "false");
    expect((await POST(request({ prompt: "Hello" }))).status).toBe(404);
  });

  it("requires a real platform session even when the public sample is enabled", async () => {
    vi.mocked(getServerAuthToken).mockResolvedValue(null);
    expect((await POST(request({ prompt: "Hello" }))).status).toBe(401);
  });

  it("rejects oversized prompts", async () => {
    expect((await POST(request({ prompt: "x".repeat(4001) }))).status).toBe(
      400,
    );
  });

  it("returns an actionable configuration error without falling back to fake output", async () => {
    const response = await POST(request({ prompt: "Show agent performance" }));
    expect(response.status).toBe(503);
    expect((await response.json()).error).toContain("not configured");
  });
});
