import { SearchEntryFilterAnyOfItem } from "@/app/api/__generated__/models/searchEntryFilterAnyOfItem";
import { CategoryKey } from "./types";

export const categories: Array<{ key: CategoryKey; name: string }> = [
  { key: SearchEntryFilterAnyOfItem.blocks, name: "Blocks" },
  {
    key: SearchEntryFilterAnyOfItem.integrations,
    name: "Integrations",
  },
  {
    key: SearchEntryFilterAnyOfItem.marketplace_agents,
    name: "Marketplace agents",
  },
  { key: SearchEntryFilterAnyOfItem.my_agents, name: "My agents" },
];
