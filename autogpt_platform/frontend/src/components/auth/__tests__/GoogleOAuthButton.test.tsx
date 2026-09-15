import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { GoogleOAuthButton } from "../GoogleOAuthButton";

describe("GoogleOAuthButton", () => {
  test("shows links to the platform terms and privacy policy", () => {
    render(<GoogleOAuthButton onClick={vi.fn()} />);

    expect(
      screen.getByText((_, element) =>
        Boolean(
          element?.textContent ===
            "By continuing with Google, you agree to the Terms of Use and acknowledge the Privacy Policy.",
        ),
      ),
    ).toBeDefined();

    expect(
      screen.getByRole("link", { name: "Terms of Use" }).getAttribute("href"),
    ).toBe("https://agpt.co/legal/platform-terms-of-use");
    expect(
      screen.getByRole("link", { name: "Privacy Policy" }).getAttribute("href"),
    ).toBe("https://agpt.co/legal/platform-privacy-policy");
  });
});
