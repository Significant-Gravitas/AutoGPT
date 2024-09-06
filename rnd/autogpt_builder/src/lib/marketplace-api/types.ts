export type ListAgentsParams = {
  page?: number;
  page_size?: number;
  name?: string;
  keyword?: string;
  category?: string;
  description?: string;
  description_threshold?: number;
  sort_by?: string;
  sort_order?: "asc" | "desc";
};

export type AddAgentRequest = {
  graph: {
    name: string;
    description: string;
    [key: string]: any;
  };
  author: string;
  keywords: string[];
  categories: string[];
};

export type Agent = {
  id: string;
  name: string;
  description: string;
  author: string;
  keywords: string[];
  categories: string[];
  version: number;
  createdAt: string; // ISO8601 datetime string
  updatedAt: string; // ISO8601 datetime string
  views: number;
  downloads: number;
};

export type AgentList = {
  agents: Agent[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
};

export type FeaturedAgentResponse = {
  agentId: string;
  featuredCategories: string[];
  createdAt: string; // ISO8601 datetime string
  updatedAt: string; // ISO8601 datetime string
  isActive: boolean;
};

export type FeaturedAgentsList = {
  agents: FeaturedAgentResponse[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
};

export type AgentDetail = Agent & {
  graph: Record<string, any>;
};

export type AgentWithRank = Agent & {
  rank: number;
};

export type AgentListResponse = AgentList;

export type AgentDetailResponse = AgentDetail;

export type AgentResponse = Agent;

export type UniqueCategoriesResponse = {
  unique_categories: string[];
};

export enum InstallationLocation {
  LOCAL = "local",
  CLOUD = "cloud",
}

export type AgentInstalledFromMarketplaceEventData = {
  marketplace_agent_id: string;
  installed_agent_id: string;
  installation_location: InstallationLocation;
};

export type AgentInstalledFromTemplateEventData = {
  template_id: string;
  installed_agent_id: string;
  installation_location: InstallationLocation;
};

export interface AgentInstalledFromMarketplaceEvent {
  event_name: "agent_installed_from_marketplace";
  event_data: AgentInstalledFromMarketplaceEventData;
}

export interface AgentInstalledFromTemplateEvent {
  event_name: "agent_installed_from_template";
  event_data: AgentInstalledFromTemplateEventData;
}

export type AnalyticsEvent =
  | AgentInstalledFromMarketplaceEvent
  | AgentInstalledFromTemplateEvent;
