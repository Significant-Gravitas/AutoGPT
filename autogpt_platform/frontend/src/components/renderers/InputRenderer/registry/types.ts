import { BlockUIType } from "@/app/(platform)/build/components/types";

export type ExtraContext = {
  nodeId?: string;
  uiType?: BlockUIType;
  size?: "small" | "medium" | "large";
};
