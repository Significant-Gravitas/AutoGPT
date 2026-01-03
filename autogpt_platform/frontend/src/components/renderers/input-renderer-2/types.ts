import { BlockUIType } from "@/lib/autogpt-server-api/types";
import { FormContextType } from "@rjsf/utils";

export interface ExtendedFormContextType extends FormContextType {
  nodeId?: string;
  uiType?: BlockUIType;
  showHandles?: boolean;
  size?: "small" | "medium" | "large";
}

export type PathSegment = {
  key: string;
  type: "property" | "item" | "additional" | "normal";
  index?: number;
};
