import type { GraphMeta as LegacyGraphMeta } from "@/lib/autogpt-server-api";
import type { GraphMeta as GeneratedGraphMeta } from "@/app/api/__generated__/models/graphMeta";

export type SubAgentUpdateInfo<T extends GraphMetaLike = GraphMetaLike> = {
  hasUpdate: boolean;
  currentVersion: number;
  latestVersion: number;
  latestGraph: T | null;
  isCompatible: boolean;
  incompatibilities: IncompatibilityInfo | null;
};

// Union type for GraphMeta that works with both legacy and new builder
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
