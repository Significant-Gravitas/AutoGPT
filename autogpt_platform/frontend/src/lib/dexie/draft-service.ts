import { db, BuilderDraft, DRAFT_EXPIRY_MS, cleanupExpiredDrafts } from "./db";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "@/app/(platform)/build/components/FlowEditor/edges/CustomEdge";
import isEqual from "lodash/isEqual";

const SESSION_TEMP_ID_KEY = "builder_temp_flow_id";

export function getOrCreateTempFlowId(): string {
  if (typeof window === "undefined") {
    return `temp_${crypto.randomUUID()}`;
  }

  let tempId = sessionStorage.getItem(SESSION_TEMP_ID_KEY);
  if (!tempId) {
    tempId = `temp_${crypto.randomUUID()}`;
    sessionStorage.setItem(SESSION_TEMP_ID_KEY, tempId);
  }
  return tempId;
}

export function clearTempFlowId(): void {
  if (typeof window !== "undefined") {
    sessionStorage.removeItem(SESSION_TEMP_ID_KEY);
  }
}

export function getTempFlowId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return sessionStorage.getItem(SESSION_TEMP_ID_KEY);
}

export interface DraftData {
  nodes: CustomNode[];
  edges: CustomEdge[];
  graphSchemas: {
    input: Record<string, unknown> | null;
    credentials: Record<string, unknown> | null;
    output: Record<string, unknown> | null;
  };
  nodeCounter: number;
  flowVersion?: number;
}

export const draftService = {
  async saveDraft(flowId: string, data: DraftData): Promise<void> {
    const draft: BuilderDraft = {
      id: flowId,
      nodes: data.nodes,
      edges: data.edges,
      graphSchemas: data.graphSchemas,
      nodeCounter: data.nodeCounter,
      savedAt: Date.now(),
      flowVersion: data.flowVersion,
    };

    await db.drafts.put(draft);
  },

  async loadDraft(flowId: string): Promise<BuilderDraft | null> {
    const draft = await db.drafts.get(flowId);

    if (!draft) {
      return null;
    }
    const age = Date.now() - draft.savedAt;
    if (age > DRAFT_EXPIRY_MS) {
      await this.deleteDraft(flowId);
      return null;
    }

    return draft;
  },

  async deleteDraft(flowId: string): Promise<void> {
    await db.drafts.delete(flowId);
  },

  async hasDraft(flowId: string): Promise<boolean> {
    const draft = await db.drafts.get(flowId);
    if (!draft) return false;

    // Check expiry
    const age = Date.now() - draft.savedAt;
    if (age > DRAFT_EXPIRY_MS) {
      await this.deleteDraft(flowId);
      return false;
    }

    return true;
  },

  isDraftDifferent(
    draft: BuilderDraft,
    currentNodes: CustomNode[],
    currentEdges: CustomEdge[],
  ): boolean {
    const cleanNodes = (nodes: CustomNode[]) =>
      nodes.map((node) => ({
        id: node.id,
        position: node.position,
        data: {
          hardcodedValues: node.data.hardcodedValues,
          title: node.data.title,
          block_id: node.data.block_id,
          metadata: node.data.metadata,
        },
      }));

    const cleanEdges = (edges: CustomEdge[]) =>
      edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      }));

    const draftNodesClean = cleanNodes(draft.nodes);
    const currentNodesClean = cleanNodes(currentNodes);
    const draftEdgesClean = cleanEdges(draft.edges);
    const currentEdgesClean = cleanEdges(currentEdges);

    const nodesDifferent = !isEqual(draftNodesClean, currentNodesClean);
    const edgesDifferent = !isEqual(draftEdgesClean, currentEdgesClean);

    return nodesDifferent || edgesDifferent;
  },

  cleanupExpired: cleanupExpiredDrafts,
};
