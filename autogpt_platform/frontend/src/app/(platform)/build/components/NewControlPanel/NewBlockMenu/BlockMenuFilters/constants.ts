import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";
import { CategoryKey } from "./types";

export const categories: Array<{ key: CategoryKey; name: string }> = [
  { key: GetV2BuilderSearchFilterAnyOfItem.blocks, name: "Blocks" },
  {
    key: GetV2BuilderSearchFilterAnyOfItem.integrations,
    name: "Integrations",
  },
  {
    key: GetV2BuilderSearchFilterAnyOfItem.marketplace_agents,
    name: "Marketplace agents",
  },
  { key: GetV2BuilderSearchFilterAnyOfItem.my_agents, name: "My agents" },
];
