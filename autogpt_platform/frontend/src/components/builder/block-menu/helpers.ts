import { Filters } from "./block-menu-provider";

export const getDefaultFilters = (): Filters => ({
  categories: {
    blocks: false,
    integrations: false,
    marketplace_agents: false,
    my_agents: false,
    providers: false,
  },
  createdBy: [],
});