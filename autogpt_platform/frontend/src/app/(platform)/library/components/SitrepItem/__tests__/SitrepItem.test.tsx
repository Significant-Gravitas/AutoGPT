import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { SitrepItem } from "../SitrepItem";
import type { SitrepItemData } from "../../../types";

const ITEM: SitrepItemData = {
  id: "run-1",
  agentID: "agent-1",
  agentName: "Daily digest",
  agentImageUrl: "https://cdn.test/agent.png",
  priority: "success",
  message: "Finished a run",
  status: "healthy" as SitrepItemData["status"],
};

describe("SitrepItem", () => {
  test("renders the agent image while the URL loads", () => {
    render(<SitrepItem item={ITEM} />);
    expect(screen.getByAltText("Daily digest")).toBeDefined();
  });

  test("falls back to the priority icon when the image URL is dead", () => {
    render(<SitrepItem item={ITEM} />);

    fireEvent.error(screen.getByAltText("Daily digest"));

    expect(screen.queryByAltText("Daily digest")).toBeNull();
    expect(screen.getByText("Daily digest")).toBeDefined();
  });
});
