import { describe, expect, it } from "vitest";
import {
  getMicrosoftSetupReason,
  microsoftCallbackMessage,
  getWorkIQAdminConsentURL,
} from "./microsoft-setup";

describe("Microsoft sign-in recovery", () => {
  it("recognizes a missing tenant service from Microsoft's callback", () => {
    expect(getMicrosoftSetupReason(new Error("AADSTS650052: missing"))).toBe(
      "tenant_setup_required",
    );
  });

  it("recognizes safe structured backend errors", () => {
    expect(
      getMicrosoftSetupReason({
        response: { detail: { reason: "admin_consent_required" } },
      }),
    ).toBe("admin_consent_required");
  });

  it("treats declined consent as a retry choice", () => {
    expect(getMicrosoftSetupReason(new Error("AADSTS65004: declined"))).toBe(
      "access_denied",
    );
  });

  it("does not expose provider descriptions or HTML in recovery copy", () => {
    const message = microsoftCallbackMessage(
      "invalid_resource",
      "AADSTS650052: secret </script><script>alert(1)</script>",
    );
    expect(message).toContain("enable Work IQ");
    expect(message).not.toContain("secret");
    expect(message).not.toContain("<script>");
  });

  it("only builds organization approval links for valid work tenants", () => {
    expect(
      getWorkIQAdminConsentURL("9188040d-6c67-4c5b-b112-36a304b66dad"),
    ).toBeNull();
    expect(getWorkIQAdminConsentURL("evil.example/")).toBeNull();
    expect(
      getWorkIQAdminConsentURL("dca42f33-178d-462f-9fea-f4e22af3a472"),
    ).toContain(
      "/dca42f33-178d-462f-9fea-f4e22af3a472/adminconsent?client_id=fdcc1f02",
    );
  });
});
