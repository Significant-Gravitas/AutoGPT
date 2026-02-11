import type {
  Graph as LegacyGraph,
  GraphMeta as LegacyGraphMeta,
} from "@/lib/autogpt-server-api";
import type { GraphModel as GeneratedGraph } from "@/app/api/__generated__/models/graphModel";
import type { GraphMeta as GeneratedGraphMeta } from "@/app/api/__generated__/models/graphMeta";

export type SubAgentUpdateInfo<T extends GraphLike = GraphLike> = {
  hasUpdate: boolean;
  currentVersion: number;
  latestVersion: number;
  latestGraph: T | null;
  isCompatible: boolean;
  incompatibilities: IncompatibilityInfo | null;
};

// Union type for Graph (with schemas) that works with both legacy and new builder
export type GraphLike = LegacyGraph | GeneratedGraph;

// Union type for GraphMeta (without schemas) for version detection
export type GraphMetaLike = LegacyGraphMeta | GeneratedGraphMeta;

export type IncompatibilityInfo = {
  missingInputs: string[]; // Connected inputs that no longer exist
  missingOutputs: string[]; // Connected outputs that no longer exist
  newInputs: string[]; // Inputs that exist in new version but not in current
  newOutputs: string[]; // Outputs that exist in new version but not in current
  newRequiredInputs: string[]; // New required inputs not in current version or not required
  inputTypeMismatches: Array<{
    name: string;
    oldType: string;
    newType: string;
  }>; // Connected inputs where the type has changed
};
