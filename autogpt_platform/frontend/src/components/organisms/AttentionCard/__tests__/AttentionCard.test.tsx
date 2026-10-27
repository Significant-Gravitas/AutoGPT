import { expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { AttentionCard } from "../AttentionCard";

test("a null density falls back to the default padding and row gap together", () => {
  render(
    <AttentionCard.Root data-testid="card" density={null}>
      <AttentionCard.Row data-testid="row">content</AttentionCard.Row>
    </AttentionCard.Root>,
  );

  expect(screen.getByTestId("card").className).toContain("p-3");
  expect(screen.getByTestId("row").className).toContain("gap-3");
});

test("a featured card pads its inner surface, not its gradient border", () => {
  render(
    <AttentionCard.Root
      data-testid="card"
      variant="featured"
      density="roomy"
      asChild
    >
      <section aria-label="Offer">offer</section>
    </AttentionCard.Root>,
  );

  const card = screen.getByTestId("card");
  expect(card.tagName).toBe("SECTION");
  expect(card.className).toContain("p-px");
  expect(card.className).not.toContain("p-5");
  expect(card.firstElementChild?.className).toContain("p-5");
});
