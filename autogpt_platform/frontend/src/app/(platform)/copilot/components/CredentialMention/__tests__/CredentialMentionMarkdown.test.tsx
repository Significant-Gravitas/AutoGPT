import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import type { ComponentProps } from "react";
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

  it("keeps the scrollable table wrapper from the message renderer", async () => {
    const { container } = render(
      <CredentialMentionMarkdown>
        {
          "| Account | Status |\n| --- | --- |\n| [Work Gmail](credential://google/work-secret-id) | ok |"
        }
      </CredentialMentionMarkdown>,
    );
    expect(await screen.findByText("Work Gmail")).toBeTruthy();
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    expect(table?.parentElement?.classList.contains("overflow-x-auto")).toBe(
      true,
    );
  });

  it("keeps the same badge and image elements while the reply streams in", async () => {
    function FramedImage({ src, alt }: ComponentProps<"img">) {
      return <span data-testid="framed-image" data-src={src} title={alt} />;
    }
    const components = { img: FramedImage };
    const reply =
      "Use [Work Gmail](credential://google/work-secret-id) with ![chart](https://example.com/chart.png)";
    const { container, rerender } = render(
      <CredentialMentionMarkdown components={components}>
        {reply}
      </CredentialMentionMarkdown>,
    );
    const framed = await screen.findByTestId("framed-image");
    const badge = container.querySelector('img[src*="google"]');
    expect(badge).not.toBeNull();

    rerender(
      <CredentialMentionMarkdown components={components}>
        {`${reply} and more text`}
      </CredentialMentionMarkdown>,
    );
    expect(await screen.findByText(/and more text/)).toBeTruthy();

    expect(screen.getByTestId("framed-image")).toBe(framed);
    expect(container.querySelector('img[src*="google"]')).toBe(badge);
  });

  it("uses an image component the caller replaced in place on the same map", async () => {
    function FirstImage({ src }: ComponentProps<"img">) {
      return <span data-testid="first-image" data-src={src} />;
    }
    function SecondImage({ src }: ComponentProps<"img">) {
      return <span data-testid="second-image" data-src={src} />;
    }
    const components: { img: typeof FirstImage } = { img: FirstImage };
    const reply = "See ![chart](https://example.com/chart.png)";
    const { rerender } = render(
      <CredentialMentionMarkdown components={components}>
        {reply}
      </CredentialMentionMarkdown>,
    );
    expect(await screen.findByTestId("first-image")).toBeTruthy();

    components.img = SecondImage;
    rerender(
      <CredentialMentionMarkdown components={components}>
        {`${reply} and more text`}
      </CredentialMentionMarkdown>,
    );

    expect(await screen.findByTestId("second-image")).toBeTruthy();
  });
});
