// Define type aliases for request and response data structures using your preferred style

export type AddAgentRequest = {
    graph: Record<string, any>;
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
};

export type AgentList = {
    agents: Agent[];
    total_count: number;
    page: number;
    page_size: number;
    total_pages: number;
};

export type AgentDetail = {
    id: string;
    name: string;
    description: string;
    author: string;
    keywords: string[];
    categories: string[];
    version: number;
    createdAt: string; // ISO8601 datetime string
    updatedAt: string; // ISO8601 datetime string
    graph: Record<string, any>;
};
