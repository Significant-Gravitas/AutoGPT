import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import MicrosoftSetupPage from "../page";

describe("Microsoft setup page", () => {
  it("says what AutoGPT asks for and how an admin approves it in either sign-in mode", () => {
    render(<MicrosoftSetupPage />);

    expect(
      screen.getByText("Lets Otto read, draft and send your email"),
    ).toBeTruthy();
    const step = screen
      .getAllByRole("listitem")
      .find((item) =>
        item.textContent?.includes("Approve for my whole organization"),
      );
    // The button exists only with browser sign-in on; sign-in codes need
    // Microsoft's own admin consent instead.
    expect(step?.textContent).toContain("sign-in codes");
    expect(
      screen
        .getByRole("link", { name: "Microsoft's guide" })
        .getAttribute("href"),
    ).toBe(
      "https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/grant-admin-consent",
    );
  });
});
