import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";

export function getNodeInputOutputSummary(node: SearchableNode) {
  if (!node?.data) return "";

  const inputs = Object.keys(node.data.inputSchema?.properties || {});
  const outputs = Object.keys(node.data.outputSchema?.properties || {});
  const parts = [];

  if (inputs.length > 0) {
    parts.push(
      `Inputs: ${inputs.slice(0, 3).join(", ")}${inputs.length > 3 ? "..." : ""}`,
    );
  }
  if (outputs.length > 0) {
    parts.push(
      `Outputs: ${outputs.slice(0, 3).join(", ")}${outputs.length > 3 ? "..." : ""}`,
    );
  }

  return parts.join(" | ");
}
