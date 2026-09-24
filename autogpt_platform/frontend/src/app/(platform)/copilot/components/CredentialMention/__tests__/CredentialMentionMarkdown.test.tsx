import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { CredentialMentionMarkdown } from "../CredentialMentionMarkdown";

describe("credential badges in messages", () => {
  it("shows names and logos in place of credential references without losing surrounding markdown", async () => {
    const { container } = render(
      <CredentialMentionMarkdown>
        {
          "Check **today** in [Work Gmail](credential://google/work-secret-id) and [Personal Gmail](credential://google/personal-secret-id)."
        }
      </CredentialMentionMarkdown>,
    );
    expect(await screen.findByText("Work Gmail")).toBeTruthy();
    expect(screen.getByText("Personal Gmail")).toBeTruthy();
    expect(screen.getByText("today").classList.contains("font-semibold")).toBe(
      true,
    );
    expect(container.textContent).not.toContain("secret-id");
    expect(container.textContent).not.toContain("credential://");
    expect(container.querySelectorAll('img[src*="google"]')).toHaveLength(2);
    expect(container.querySelectorAll("a")).toHaveLength(0);
  });
});
