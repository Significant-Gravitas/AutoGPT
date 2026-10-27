import { describe, expect, it } from "vitest";
import { GET } from "../route";

describe("Microsoft callback recovery", () => {
  it("returns a safe, state-bound setup error instead of losing Microsoft's reason", async () => {
    const response = await GET(
      new Request(
        "https://app.example/auth/integrations/oauth_callback?state=flow-state&error=invalid_resource&error_description=AADSTS650052%3A%20private-upstream-text",
      ),
    );
    const html = await response.text();
    expect(html).toContain("flow-state");
    expect(html).toContain("AADSTS650052");
    expect(html).toContain("enable Work IQ");
    expect(html).not.toContain("private-upstream-text");
    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("confirms an organization-wide approval, which has no code to exchange", async () => {
    const response = await GET(
      new Request(
        "https://app.example/auth/integrations/oauth_callback?admin_consent=True&tenant=company",
      ),
    );
    const html = await response.text();
    expect(html).toContain("Approved for your organization");
    expect(html).not.toContain("oauth_popup_result");
  });
});
