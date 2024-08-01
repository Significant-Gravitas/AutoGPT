import { Graph, Block, Node } from "./types";

/** Creates a copy of the graph with all secrets removed */
export function safeCopyGraph(graph: Graph, block_defs: Block[]): Graph {
  return {
    ...graph,
    nodes: graph.nodes.map(node => {
      const block = block_defs.find(b => b.id == node.block_id)!;
      return {
        ...node,
        input_default: Object.keys(node.input_default)
          .filter(k => !block.inputSchema.properties[k].secret)
          .reduce((obj: Node['input_default'], key) => {
            obj[key] = node.input_default[key];
            return obj;
          }, {}),
      }
    }),
  }
}
