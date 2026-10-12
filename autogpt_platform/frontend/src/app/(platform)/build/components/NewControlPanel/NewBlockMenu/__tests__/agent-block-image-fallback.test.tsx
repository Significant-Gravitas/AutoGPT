import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { MarketplaceAgentBlock } from "../MarketplaceAgentBlock";
import { UGCAgentBlock } from "../UGCAgentBlock";

describe("MarketplaceAgentBlock", () => {
  test("drops a dead image URL and keeps the title", () => {
    render(
      <MarketplaceAgentBlock
        title="Daily digest"
        image_url="https://cdn.test/dead.png"
        slug="digest"
        loading={false}
      />,
    );

    fireEvent.error(screen.getByAltText("integration-icon"));

    expect(screen.queryByAltText("integration-icon")).toBeNull();
    expect(screen.getByText("Daily digest")).toBeDefined();
  });
});

describe("UGCAgentBlock", () => {
  test("drops a dead image URL and keeps the title", () => {
    render(
      <UGCAgentBlock title="My agent" image_url="https://cdn.test/dead.png" />,
    );

    fireEvent.error(screen.getByAltText("integration-icon"));

    expect(screen.queryByAltText("integration-icon")).toBeNull();
    expect(screen.getByText("My agent")).toBeDefined();
  });
});
