import { describe, expect, it } from "vitest";
import {
  checkoutPresentation,
  continueAfterDecision,
  isLinkCheckoutOutput,
  purchaseHost,
} from "./helpers";

describe("Link checkout presentation", () => {
  it("shows a trusted verification action and Link's message for it", () => {
    const output = checkoutPresentation({
      status: "requires_action",
      action_url: "https://app.link.com/verify/test",
      action_message: "Your bank needs to confirm it's you.",
    });
    expect(output.approvalUrl).toBe("https://app.link.com/verify/test");
    expect(output.linkLabel).toBe("Complete verification in Link");
    expect(output.actionMessage).toBe("Your bank needs to confirm it's you.");
  });

  it("shows a hosted Link approval only while waiting for approval", () => {
    const output = {
      status: "pending_approval",
      approval_url: "https://app.link.com/activity/approve/lsrq_test",
    };
    expect(checkoutPresentation(output).approvalUrl).toBe(output.approval_url);
    expect(
      checkoutPresentation({ ...output, status: "submitted" }).approvalUrl,
    ).toBe("");
  });

  it.each([
    "javascript:alert(1)",
    "https://app.link.com.evil.example/approve",
    "https://evil.example/app.link.com",
    "https://user@app.link.com/approve",
    "http://app.link.com/approve",
    "https://app.link.com:8443/approve",
  ])("rejects an untrusted approval destination: %s", (approval_url) => {
    expect(
      checkoutPresentation({ status: "pending_approval", approval_url })
        .approvalUrl,
    ).toBe("");
  });

  it("accepts Stripe-hosted verification pages", () => {
    expect(
      checkoutPresentation({
        status: "requires_action",
        action_url: "https://hooks.stripe.com/3d_secure/abc",
      }).approvalUrl,
    ).toBe("https://hooks.stripe.com/3d_secure/abc");
  });

  it("asks in the chat only for an in-chat approval that is still waiting", () => {
    expect(
      checkoutPresentation({
        approval_mode: "in_app",
        status: "awaiting_approval",
      }).approvesInChat,
    ).toBe(true);
    expect(
      checkoutPresentation({
        approval_mode: "link",
        status: "pending_approval",
      }).approvesInChat,
    ).toBe(false);
    expect(
      checkoutPresentation({ approval_mode: "in_app", status: "submitted" })
        .approvesInChat,
    ).toBe(false);
  });

  it("does not describe browser submission as confirmed payment", () => {
    expect(checkoutPresentation({ status: "submitted" }).status).toContain(
      "confirmation pending",
    );
  });

  it("formats smallest currency units correctly", () => {
    expect(
      checkoutPresentation({ amount: 1234, currency: "usd" }).total,
    ).toContain("12.34");
    expect(
      checkoutPresentation({ amount: 1234, currency: "jpy" }).total,
    ).toContain("1,234");
  });
});

describe("continueAfterDecision", () => {
  it("names the exact checkout to complete after an approval", () => {
    const message = continueAfterDecision({
      checkoutId: "a".repeat(32),
      approved: true,
    });
    expect(message).toContain(`I approved purchase ${"a".repeat(32)}`);
    expect(message).toContain(`{"checkout_id":"${"a".repeat(32)}"}`);
  });

  it("tells the agent not to buy after a decline", () => {
    const message = continueAfterDecision({
      checkoutId: "a".repeat(32),
      approved: false,
    });
    expect(message).toContain("Don't buy it");
    expect(message).not.toContain("browser_complete_link_payment");
  });
});

describe("isLinkCheckoutOutput", () => {
  it("accepts a checkout only from the checkout tools", () => {
    const output = { type: "browser_checkout" };
    expect(isLinkCheckoutOutput("browser_request_link_payment", output)).toBe(
      true,
    );
    expect(isLinkCheckoutOutput("web_fetch", output)).toBe(false);
    expect(isLinkCheckoutOutput(undefined, output)).toBe(false);
    expect(
      isLinkCheckoutOutput("browser_link_payment_status", { type: "error" }),
    ).toBe(false);
  });
});

describe("purchaseHost", () => {
  it("shows the site the card is used on, and nothing for a bad URL", () => {
    expect(purchaseHost("https://shop.example/checkout")).toBe("shop.example");
    expect(purchaseHost("not a url")).toBe("");
  });
});
