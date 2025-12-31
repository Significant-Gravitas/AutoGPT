import Dexie, { type EntityTable } from "dexie";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "@/app/(platform)/build/components/FlowEditor/edges/CustomEdge";

// 24 hrs expiry
export const DRAFT_EXPIRY_MS = 24 * 60 * 60 * 1000;

export interface BuilderDraft {
  id: string;
  nodes: CustomNode[];
  edges: CustomEdge[];
  graphSchemas: {
    input: Record<string, unknown> | null;
    credentials: Record<string, unknown> | null;
    output: Record<string, unknown> | null;
  };
  nodeCounter: number;
  savedAt: number;
  flowVersion?: number;
}

class BuilderDatabase extends Dexie {
  drafts!: EntityTable<BuilderDraft, "id">;

  constructor() {
    super("AutoGPTBuilderDB");

    this.version(1).stores({
      drafts: "id, savedAt",
    });
  }
}

// Singleton database instance
export const db = new BuilderDatabase();

export async function cleanupExpiredDrafts(): Promise<number> {
  const expiryThreshold = Date.now() - DRAFT_EXPIRY_MS;

  const deletedCount = await db.drafts
    .where("savedAt")
    .below(expiryThreshold)
    .delete();

  return deletedCount;
}
