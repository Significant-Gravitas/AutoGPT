import { MarkerType } from "@xyflow/react";
import { type Dispatch, type SetStateAction } from "react";
import { useEdgeStore } from "../../stores/edgeStore";
import { useNodeStore } from "../../stores/nodeStore";
import type { CustomEdge } from "../FlowEditor/edges/CustomEdge";
import type { CustomNode } from "../FlowEditor/nodes/CustomNode/CustomNode";
import { GraphAction, getActionKey, getNodeDisplayName } from "./helpers";
import type { useToast } from "@/components/molecules/Toast/use-toast";

export type ToastFn = ReturnType<typeof useToast>["toast"];

/** Maximum number of undo entries to keep. Oldest entries are dropped when the limit is reached. */
export const MAX_UNDO = 20;

/** Keys that must never be written via `update_node_input` to prevent prototype pollution. */
const DANGEROUS_KEYS = new Set(["__proto__", "constructor", "prototype"]);

/**
 * Default edge arrowhead color. Mirrors the value used by the manual
 * addEdge helper in `edgeStore` so chat-applied edges render identically.
 */
export const DEFAULT_EDGE_MARKER_COLOR = "#555";

/**
 * Deep-clone an array of simple objects. Prefers `structuredClone` when
 * available (isolates nested data from later in-place mutation) and falls
 * back to an element-level spread on older environments.
 *
 * Used for undo snapshots where holding the original object graph keeps the
 * restore state independent of subsequent store mutations.
 */
export function safeCloneArray<T extends object>(items: T[]): T[] {
  if (typeof structuredClone === "function") {
    try {
      return structuredClone(items);
    } catch {
      // Fall through — some items may contain non-cloneable values
      // (functions, DOM nodes, class instances). A shallow spread is the
      // best we can do on the fallback path.
    }
  }
  return items.map((item) => ({ ...item }));
}

/** Snapshot of node data taken before an action is applied, enabling undo. */
export interface UndoSnapshot {
  actionKey: string;
  restore: () => void;
}

/**
 * Push a new undo snapshot onto the stack, trimming the oldest entry when at
 * the `MAX_UNDO` cap. Extracted to keep the action-apply branches DRY.
 */
export function pushUndoEntry(
  setUndoStack: Dispatch<SetStateAction<UndoSnapshot[]>>,
  entry: UndoSnapshot,
): void {
  setUndoStack((prev) => {
    const trimmed = prev.length >= MAX_UNDO ? prev.slice(1) : prev;
    return [...trimmed, entry];
  });
}

/**
 * Deep-clones a nodes array so an undo snapshot is isolated from in-place
 * mutations of node data elsewhere in the app. Uses `safeCloneArray` with a
 * node-specific fallback that also copies the `data` sub-object so the
 * shallow path still isolates the field commonly mutated by the builder.
 */
export function cloneNodes(nodes: CustomNode[]): CustomNode[] {
  if (typeof structuredClone === "function") {
    try {
      return structuredClone(nodes);
    } catch {
      // Fall through to shallow copy — some nodes may contain non-cloneable values.
    }
  }
  return nodes.map((n) => ({ ...n, data: { ...n.data } }));
}

/** Removes an applied action key from the set — used by undo restore callbacks. */
function removeAppliedActionKey(
  setAppliedActionKeys: Dispatch<SetStateAction<Set<string>>>,
  key: string,
): void {
  setAppliedActionKeys((keys) => {
    const next = new Set(keys);
    next.delete(key);
    return next;
  });
}

export interface ApplyActionDeps {
  toast: ToastFn;
  setNodes: (nodes: CustomNode[]) => void;
  setEdges: (edges: CustomEdge[]) => void;
  setUndoStack: Dispatch<SetStateAction<UndoSnapshot[]>>;
  setAppliedActionKeys: Dispatch<SetStateAction<Set<string>>>;
}

/**
 * Applies an `update_node_input` action to the node store, returning `true` on
 * success and `false` when validation fails (node missing, invalid key, etc).
 * All mutations go through `setNodes` so they bypass the global history store
 * and stay independent of the builder's Ctrl+Z stack.
 */
export function applyUpdateNodeInput(
  action: Extract<GraphAction, { type: "update_node_input" }>,
  deps: ApplyActionDeps,
): boolean {
  const { toast, setNodes, setUndoStack, setAppliedActionKeys } = deps;
  // Read live state for both validation and mutation so rapid successive
  // applies see the latest nodes rather than a stale render-cycle snapshot.
  const liveNodes = useNodeStore.getState().nodes;
  const node = liveNodes.find((n) => n.id === action.nodeId);
  if (!node) {
    toast({
      title: "Cannot apply change",
      description: `Node "${action.nodeId}" was not found in the graph.`,
      variant: "destructive",
    });
    return false;
  }
  // Block prototype-polluting keys regardless of schema presence.
  if (DANGEROUS_KEYS.has(action.key)) {
    toast({
      title: "Cannot apply change",
      description: `Field "${action.key}" is not a valid input.`,
      variant: "destructive",
    });
    return false;
  }
  // Reject keys not present in the node's input schema to prevent writing
  // arbitrary fields that the block does not support.
  const schemaProps = node.data.inputSchema?.properties;
  if (
    schemaProps &&
    !Object.prototype.hasOwnProperty.call(schemaProps, action.key)
  ) {
    toast({
      title: "Cannot apply change",
      description: `Field "${action.key}" is not a valid input for "${getNodeDisplayName(node, node.id)}".`,
      variant: "destructive",
    });
    return false;
  }
  // Snapshot only the single field that is about to change so the undo
  // restore can revert it without clobbering unrelated edits the user may
  // have made to other nodes (or to other fields on this node) in between.
  const hadKey = Object.prototype.hasOwnProperty.call(
    node.data.hardcodedValues ?? {},
    action.key,
  );
  const prevFieldValue: unknown = hadKey
    ? (node.data.hardcodedValues as Record<string, unknown>)[action.key]
    : undefined;
  const nextNodes = liveNodes.map((n) =>
    n.id === action.nodeId
      ? {
          ...n,
          data: {
            ...n.data,
            hardcodedValues: {
              ...n.data.hardcodedValues,
              [action.key]: action.value,
            },
          },
        }
      : n,
  );
  const key = getActionKey(action);
  pushUndoEntry(setUndoStack, {
    actionKey: key,
    restore: () => {
      // Differential restore: re-read the live nodes at undo time and only
      // revert `action.key` on the target node. This preserves any other
      // edits (to this node or other nodes) that happened after apply.
      const currentNodes = useNodeStore.getState().nodes;
      // If the target node was deleted between apply and undo, skip the
      // restore and notify the user so they aren't left wondering why nothing
      // changed. The stale undo entry is still popped by the caller.
      if (!currentNodes.some((n) => n.id === action.nodeId)) {
        toast({
          title: "Undo skipped",
          description: `Node "${action.nodeId}" no longer exists in the graph.`,
          variant: "destructive",
        });
        removeAppliedActionKey(setAppliedActionKeys, key);
        return;
      }
      const restoredNodes = currentNodes.map((n) => {
        if (n.id !== action.nodeId) return n;
        const { [action.key]: _omitted, ...rest } = (n.data.hardcodedValues ??
          {}) as Record<string, unknown>;
        void _omitted;
        const nextHardcoded = hadKey
          ? { ...rest, [action.key]: prevFieldValue }
          : rest;
        return { ...n, data: { ...n.data, hardcodedValues: nextHardcoded } };
      });
      setNodes(restoredNodes);
      removeAppliedActionKey(setAppliedActionKeys, key);
    },
  });
  setNodes(nextNodes);
  return true;
}

/**
 * Applies a `connect_nodes` action to the edge store. Returns `true` on
 * success (or on idempotent no-op when the edge already exists) and `false`
 * when validation fails.
 */
export function applyConnectNodes(
  action: Extract<GraphAction, { type: "connect_nodes" }>,
  deps: ApplyActionDeps,
): boolean {
  const { toast, setEdges, setUndoStack, setAppliedActionKeys } = deps;
  // Read live state so validation reflects the current graph even when
  // multiple actions are applied within the same render cycle.
  const liveNodes = useNodeStore.getState().nodes;
  const sourceNode = liveNodes.find((n) => n.id === action.source);
  const targetNode = liveNodes.find((n) => n.id === action.target);
  if (!sourceNode || !targetNode) {
    toast({
      title: "Cannot apply connection",
      description: `One or both nodes (${action.source}, ${action.target}) were not found.`,
      variant: "destructive",
    });
    return false;
  }
  // Validate that the referenced handles exist on the respective nodes.
  const srcProps = sourceNode.data.outputSchema?.properties;
  const tgtProps = targetNode.data.inputSchema?.properties;
  if (
    srcProps &&
    !Object.prototype.hasOwnProperty.call(srcProps, action.sourceHandle)
  ) {
    toast({
      title: "Cannot apply connection",
      description: `Output handle "${action.sourceHandle}" does not exist on "${getNodeDisplayName(sourceNode, action.source)}".`,
      variant: "destructive",
    });
    return false;
  }
  if (
    tgtProps &&
    !Object.prototype.hasOwnProperty.call(tgtProps, action.targetHandle)
  ) {
    toast({
      title: "Cannot apply connection",
      description: `Input handle "${action.targetHandle}" does not exist on "${getNodeDisplayName(targetNode, action.target)}".`,
      variant: "destructive",
    });
    return false;
  }
  const edgeId = `${action.source}:${action.sourceHandle}->${action.target}:${action.targetHandle}`;
  const liveEdges = useEdgeStore.getState().edges;
  // Guard against duplicate edges — the same connection may appear after an
  // undo-then-reapply or from identical suggestions across AI messages.
  const alreadyExists = liveEdges.some(
    (e) =>
      e.source === action.source &&
      e.target === action.target &&
      e.sourceHandle === action.sourceHandle &&
      e.targetHandle === action.targetHandle,
  );
  if (alreadyExists) {
    // Edge already present — caller (handleApplyAction) will mark as applied.
    return true;
  }
  const key = getActionKey(action);
  pushUndoEntry(setUndoStack, {
    actionKey: key,
    restore: () => {
      // Differential restore: re-read the live edges at undo time and only
      // remove the specific edge that this action added. This preserves any
      // other edges (added manually or by later AI actions) that may have
      // been created after apply.
      const currentEdges = useEdgeStore.getState().edges;
      const restoredEdges = currentEdges.filter(
        (e) =>
          !(
            e.source === action.source &&
            e.target === action.target &&
            e.sourceHandle === action.sourceHandle &&
            e.targetHandle === action.targetHandle
          ),
      );
      setEdges(restoredEdges);
      removeAppliedActionKey(setAppliedActionKeys, key);
    },
  });
  setEdges([
    ...liveEdges,
    {
      id: edgeId,
      source: action.source,
      target: action.target,
      sourceHandle: action.sourceHandle,
      targetHandle: action.targetHandle,
      type: "custom",
      // Match the markerEnd style used by addEdge in edgeStore so
      // chat-applied edges render with the same arrowhead as manually drawn ones.
      markerEnd: {
        type: MarkerType.ArrowClosed,
        strokeWidth: 2,
        color: DEFAULT_EDGE_MARKER_COLOR,
      },
    },
  ]);
  return true;
}
