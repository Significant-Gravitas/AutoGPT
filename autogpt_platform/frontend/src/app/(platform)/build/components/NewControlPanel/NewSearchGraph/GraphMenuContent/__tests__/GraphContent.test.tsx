import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { BlockUIType } from "../../../../types";
import type { SearchableNode } from "../../GraphMenuSearchBar/useGraphMenuSearchBar";
import { GraphSearchContent } from "../GraphContent";

function makeNode(
  id: string,
  title: string,
  inputs: string[],
  outputs: string[],
  customizedName?: string,
): SearchableNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title,
      description: `${title} description`,
      inputSchema: {
        type: "object",
        properties: Object.fromEntries(inputs.map((n) => [n, {}])),
      },
      outputSchema: {
        type: "object",
        properties: Object.fromEntries(outputs.map((n) => [n, {}])),
      },
      uiType: BlockUIType.STANDARD,
      block_id: `block-${id}`,
      costs: [],
      categories: [],
      metadata: customizedName ? { customized_name: customizedName } : {},
    },
  };
}

function renderContent(query: string, nodes: SearchableNode[]) {
  const onNodeSelect = vi.fn();
  render(
    <GraphSearchContent
      searchQuery={query}
      onSearchChange={vi.fn()}
      filteredNodes={nodes}
      onNodeSelect={onNodeSelect}
    />,
  );
  return { onNodeSelect };
}

describe("GraphSearchContent", () => {
  test("asks for a query before anything is typed", () => {
    renderContent("", []);
    expect(screen.getByText("Start typing to search nodes")).toBeDefined();
    expect(screen.queryByText(/^Found/)).toBeNull();
  });

  test("says when nothing matches", () => {
    renderContent("zzz", []);
    expect(screen.getByText("Found 0 nodes")).toBeDefined();
    expect(
      screen.getByText("No nodes found matching your search"),
    ).toBeDefined();
  });

  test("lists matches with their type chip and summary, and selects on click", () => {
    const { onNodeSelect } = renderContent("email", [
      makeNode("3", "SendEmailBlock", ["to_email"], ["status"]),
      makeNode(
        "4",
        "SendEmailBlock",
        ["to_email"],
        ["status"],
        "Team update email",
      ),
    ]);
    expect(screen.getByText("Found 2 nodes")).toBeDefined();
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(2);
    // the customised node shows its name and a type chip
    expect(screen.getByText("Team update email")).toBeDefined();
    expect(screen.getAllByText("Send Email").length).toBeGreaterThan(0);
    fireEvent.click(options[1]);
    expect(onNodeSelect).toHaveBeenCalledWith("4");
  });
});
