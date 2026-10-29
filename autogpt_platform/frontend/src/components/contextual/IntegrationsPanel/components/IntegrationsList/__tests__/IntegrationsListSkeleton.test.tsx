import { describe, expect, test } from "vitest";

import { render, screen, within } from "@/tests/integrations/test-utils";

import { IntegrationsListSkeleton } from "../IntegrationsListSkeleton";

describe("IntegrationsListSkeleton", () => {
  test("renders 3 placeholder accordion shells with aria-busy", () => {
    render(<IntegrationsListSkeleton />);
    const root = screen.getByLabelText(/loading integrations/i);
    expect(root.getAttribute("aria-busy")).toBe("true");
    expect(
      within(root).getAllByTestId("integration-skeleton-item"),
    ).toHaveLength(3);
  });
});
