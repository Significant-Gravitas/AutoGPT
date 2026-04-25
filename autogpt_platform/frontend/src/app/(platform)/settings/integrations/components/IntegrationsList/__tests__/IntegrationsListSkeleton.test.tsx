import { describe, expect, test } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

import { IntegrationsListSkeleton } from "../IntegrationsListSkeleton";

describe("IntegrationsListSkeleton", () => {
  test("renders 3 placeholder accordion shells with aria-busy", () => {
    const { container } = render(<IntegrationsListSkeleton />);
    const root = screen.getByLabelText(/loading integrations/i);
    expect(root.getAttribute("aria-busy")).toBe("true");
    // 3 accordion shells (matches the 3-iteration map in the component).
    expect(container.querySelectorAll("div.rounded-lg.border").length).toBe(3);
  });
});
