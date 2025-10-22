import { Graph } from "@/app/api/__generated__/models/graph";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { Link } from "@/app/api/__generated__/models/link";
import { NodeModel } from "@/app/api/__generated__/models/nodeModel";
import { Node } from "@/app/api/__generated__/models/node";
import { deepEquals } from "@rjsf/utils";

export const graphsEquivalent = (
  saved: GraphModel | undefined,
  current: Graph | undefined,
): boolean => {
  if (!saved || !current) {
    return false;
  }
  const sortNodes = (nodes: NodeModel[] | Node[]) =>
    nodes.toSorted((a, b) => a.id?.localeCompare(b.id ?? "") ?? 0);

  const sortLinks = (links: Link[]) =>
    links.toSorted(
      (a, b) =>
        8 * a.source_id.localeCompare(b.source_id) +
        4 * a.sink_id.localeCompare(b.sink_id) +
        2 * a.source_name.localeCompare(b.source_name) +
        a.sink_name.localeCompare(b.sink_name),
    );
  const _saved = {
    name: saved.name,
    description: saved.description,
    nodes: sortNodes(saved.nodes ?? []).map((v) => ({
      block_id: v.block_id,
      input_default: v.input_default,
      metadata: v.metadata,
    })),
    links: sortLinks(saved.links ?? []).map((v) => ({
      sink_name: v.sink_name,
      source_name: v.source_name,
    })),
  };

  // Normalize current graph - exclude IDs
  const _current = {
    name: current.name,
    description: current.description,
    nodes: sortNodes(current.nodes ?? []).map(({ id: _, ...rest }) => rest),
    links: sortLinks(current.links ?? []).map(
      ({ source_id: _, sink_id: __, ...rest }) => rest,
    ),
  };

  return deepEquals(_saved, _current);
};
