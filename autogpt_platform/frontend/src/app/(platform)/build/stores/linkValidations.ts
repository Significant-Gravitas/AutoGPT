import { Link } from "@/app/api/__generated__/models/link";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

/**
 * Filter out edges that reference non-existent nodes.
 * Used before sending edges to the backend during save.
 */
export function filterValidEdges(
  edges: CustomEdge[],
  nodeIds: Set<string>,
): CustomEdge[] {
  return edges.filter((edge) => {
    const isValid = nodeIds.has(edge.source) && nodeIds.has(edge.target);
    if (!isValid) {
      console.warn(
        `[linkValidations] Filtering out invalid edge during save: source=${edge.source}, target=${edge.target}`,
      );
    }
    return isValid;
  });
}

/**
 * Filter out links that reference non-existent nodes.
 * Used when loading links from the backend to prevent orphan edges.
 */
export function filterValidLinks(links: Link[], nodeIds: Set<string>): Link[] {
  return links.filter((link) => {
    const isValid = nodeIds.has(link.source_id) && nodeIds.has(link.sink_id);
    if (!isValid) {
      console.warn(
        `[linkValidations] Skipping invalid link: source=${link.source_id}, sink=${link.sink_id} - node(s) not found`,
      );
    }
    return isValid;
  });
}
