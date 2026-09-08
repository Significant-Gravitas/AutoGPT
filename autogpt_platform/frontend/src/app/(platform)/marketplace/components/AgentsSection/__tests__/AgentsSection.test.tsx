import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { AgentsSection } from "../AgentsSection";

function agent(name: string, verified: boolean): StoreAgent {
  return {
    slug: name.toLowerCase(),
    agent_name: name,
    agent_image: "",
    creator: "creator",
    creator_avatar: "",
    sub_heading: "",
    description: "",
    runs: 1,
    rating: 5,
    // Empty keeps the add-to-library button, and its auth, off the card.
    agent_graph_id: "",
    verified,
  };
}

describe("AgentsSection", () => {
  test("shows each listing's verification state on its card", () => {
    render(
      <AgentsSection
        agents={[agent("Reviewed", true), agent("Unreviewed", false)]}
      />,
    );

    const cards = screen.getAllByTestId("store-card");
    const reviewed = cards.filter((c) => within(c).queryByText("Reviewed"));
    const unreviewed = cards.filter((c) => within(c).queryByText("Unreviewed"));

    expect(reviewed.length).toBeGreaterThan(0);
    expect(unreviewed.length).toBeGreaterThan(0);
    reviewed.forEach((c) =>
      expect(within(c).getByText("Verified")).toBeTruthy(),
    );
    unreviewed.forEach((c) =>
      expect(within(c).getByText("Community")).toBeTruthy(),
    );
  });
});
