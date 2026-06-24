import { IncompatibilityInfo } from "../hooks/useSubAgentUpdate/types";

export type NodeResolutionData = {
  incompatibilities: IncompatibilityInfo;
  pendingUpdate: {
    input_schema: Record<string, unknown>;
    output_schema: Record<string, unknown>;
  };
  currentSchema: {
    input_schema: Record<string, unknown>;
    output_schema: Record<string, unknown>;
  };
  pendingHardcodedValues: Record<string, unknown>;
};
