import { describe, expect, it } from "vitest";

import { GET } from "../route";

const origin = "http://localhost:3000";

function embeddedMessage(html: string): Record<string, unknown> {
  const match = html.match(/var msg = (\{.*?\});/);
  if (!match) throw new Error("callback page does not embed a message");
  return JSON.parse(match[1]);
}

describe("MCP OAuth callback route", () => {
  it("forwards code, state and the RFC 9207 issuer to the opener", async () => {
    const response = await GET(
      new Request(
        `${origin}/auth/integrations/mcp_callback?code=abc&state=xyz&iss=${encodeURIComponent("https://auth.example.com")}`,
      ),
    );
    const html = await response.text();

    expect(response.headers.get("Content-Type")).toBe("text/html");
    expect(embeddedMessage(html)).toEqual({
      success: true,
      code: "abc",
      state: "xyz",
      iss: "https://auth.example.com",
    });
    expect(html).toContain('message_type: "mcp_oauth_result"');
  });

  it("reports a null issuer when the authorization server sends none", async () => {
    const response = await GET(
      new Request(
        `${origin}/auth/integrations/mcp_callback?code=abc&state=xyz`,
      ),
    );

    expect(embeddedMessage(await response.text())).toEqual({
      success: true,
      code: "abc",
      state: "xyz",
      iss: null,
    });
  });

  it("reports failure when code or state is missing", async () => {
    const response = await GET(
      new Request(`${origin}/auth/integrations/mcp_callback?state=xyz`),
    );

    const message = embeddedMessage(await response.text());
    expect(message.success).toBe(false);
    expect(message.message).toContain("Missing parameters");
  });

  it("escapes script-breaking characters in query values", async () => {
    const response = await GET(
      new Request(
        `${origin}/auth/integrations/mcp_callback?code=${encodeURIComponent("</script><script>alert(1)</script>")}&state=xyz`,
      ),
    );

    const html = await response.text();
    expect(html).not.toContain("</script><script>alert(1)");
    expect(embeddedMessage(html).code).toBe(
      "</script><script>alert(1)</script>",
    );
  });
});
