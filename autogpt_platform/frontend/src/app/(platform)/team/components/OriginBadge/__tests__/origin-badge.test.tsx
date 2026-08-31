import { render, screen } from "@/tests/integrations/test-utils";
import { expect, test } from "vitest";
import { OriginBadge } from "../OriginBadge";

test.each([
  ["USER", "You"],
  ["SCHEDULE", "Schedule"],
  ["DREAM", "Proactive"],
  ["EXPERT", "Expert"],
])("renders %s as %s", (type, label) => {
  render(<OriginBadge createdByType={type} />);
  expect(screen.getByText(label)).toBeTruthy();
});

test("renders nothing for a missing or unknown origin", () => {
  const { container } = render(<OriginBadge createdByType={undefined} />);
  expect(container.textContent).toBe("");

  const unknown = render(<OriginBadge createdByType="SOMETHING_NEW" />);
  expect(unknown.container.textContent).toBe("");
});
