import { FilterType } from "@/app/api/__generated__/models/filterType";
import { CategoryKey } from "./types";

export const categories: Array<{ key: CategoryKey; name: string }> = [
  { key: FilterType.blocks, name: "Blocks" },
  {
    key: FilterType.integrations,
    name: "Integrations",
  },
  {
    key: FilterType.marketplace_agents,
    name: "Marketplace agents",
  },
  { key: FilterType.my_agents, name: "My agents" },
];
