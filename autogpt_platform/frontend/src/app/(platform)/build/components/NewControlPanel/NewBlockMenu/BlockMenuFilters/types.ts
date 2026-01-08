import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";

export type DefaultStateType =
  | "suggestion"
  | "all_blocks"
  | "input_blocks"
  | "action_blocks"
  | "output_blocks"
  | "integrations"
  | "marketplace_agents"
  | "my_agents";

export type CategoryKey = GetV2BuilderSearchFilterAnyOfItem;

export interface Filters {
  categories: {
    blocks: boolean;
    integrations: boolean;
    marketplace_agents: boolean;
    my_agents: boolean;
    providers: boolean;
  };
  createdBy: string[];
}

export type CategoryCounts = Record<CategoryKey, number>;
