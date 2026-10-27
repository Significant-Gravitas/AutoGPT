import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { fn } from "storybook/test";
import type { CustomNode } from "../../../FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "../../../types";
import { useGraphSearch } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { GraphSearchContent } from "./GraphContent";

function makeNode(
  id: string,
  title: string,
  description: string,
  inputs: string[],
  outputs: string[],
  customizedName?: string,
): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title,
      description,
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

const NODES: CustomNode[] = [
  makeNode("1", "AgentInputBlock", "Collects the topic", ["name"], ["result"]),
  makeNode(
    "2",
    "AITextGeneratorBlock",
    "Writes the summary",
    ["prompt", "model"],
    ["response"],
    "Morning summary",
  ),
  makeNode(
    "3",
    "SendEmailBlock",
    "Emails the summary",
    ["to_email", "subject", "body"],
    ["status"],
  ),
  makeNode(
    "4",
    "SendEmailBlock",
    "Emails the team",
    ["to_email", "subject", "body"],
    ["status"],
    "Team update email",
  ),
  makeNode("5", "AgentOutputBlock", "Returns the result", ["value"], []),
];

interface DemoProps {
  nodes: CustomNode[];
  initialQuery: string;
  onNodeSelect: (nodeID: string) => void;
}

// Feeds the content through the same search hook the graph menu uses, so the
// results and the "Found N" line come from the real scoring.
function GraphContentDemo({ nodes, initialQuery, onNodeSelect }: DemoProps) {
  const { searchQuery, setSearchQuery, filteredNodes } = useGraphSearch(nodes);
  const [seeded, setSeeded] = useState(false);

  if (!seeded) {
    setSeeded(true);
    setSearchQuery(initialQuery);
  }

  return (
    <div className="h-[32rem] w-full max-w-[46.625rem] overflow-hidden rounded-[1rem] bg-white shadow-[0_2px_6px_0_rgba(0,0,0,0.05)]">
      <GraphSearchContent
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        filteredNodes={filteredNodes}
        onNodeSelect={onNodeSelect}
      />
    </div>
  );
}

const meta = {
  title: "Build/GraphContent",
  component: GraphContentDemo,
  args: { nodes: NODES, initialQuery: "", onNodeSelect: fn() },
} satisfies Meta<typeof GraphContentDemo>;

export default meta;
type Story = StoryObj<typeof meta>;

/** No query yet: every node is listed and the first one is highlighted. */
export const AllNodes: Story = {};

/** A query with matches: the count line and the matching nodes. */
export const WithResults: Story = {
  args: { initialQuery: "email" },
};

/** A query with no matches. */
export const NoResults: Story = {
  args: { initialQuery: "zzz" },
};

/** A workflow with no nodes. */
export const EmptyWorkflow: Story = {
  args: { nodes: [] },
};
