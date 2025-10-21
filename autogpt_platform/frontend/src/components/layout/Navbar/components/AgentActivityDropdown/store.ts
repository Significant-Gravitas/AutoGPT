import { create } from "zustand";
import * as Sentry from "@sentry/nextjs";
import { storage, Key } from "@/services/storage/local-storage";
import {
  getV2ListLibraryAgents,
  type getV2ListLibraryAgentsResponse,
} from "@/app/api/__generated__/endpoints/library/library";

type AgentInfo = {
  name: string;
  description: string;
  library_agent_id?: string;
  graph_id?: string | null;
};

type AgentStore = {
  agents: AgentInfo[];
  lastUpdatedAt?: number;
  isRefreshing: boolean;
  error?: unknown;
  loadFromCache: () => void;
  refreshAll: () => Promise<void>;
};

type CachedAgents = {
  agents: AgentInfo[];
  lastUpdatedAt: number;
};

async function fetchAllLibraryAgents(): Promise<AgentInfo[]> {
  const pageSize = 100;
  let page = 1;
  const all: AgentInfo[] = [];

  // Fetch first page to get pagination info
  let res: getV2ListLibraryAgentsResponse | undefined;
  try {
    res = await getV2ListLibraryAgents({ page, page_size: pageSize });
  } catch (err) {
    Sentry.captureException(err, { tags: { context: "library_agents_fetch" } });
    throw err;
  }

  if (!res || res.status !== 200) return all;

  const { agents, pagination } = res.data;
  all.push(
    ...agents.map(
      (a): AgentInfo => ({
        name:
          a.name || (a.graph_id ? `Agent ${a.graph_id.slice(0, 8)}` : "Agent"),
        description: a.description || "",
        library_agent_id: a.id,
        graph_id: a.graph_id,
      }),
    ),
  );

  const totalPages = pagination?.total_pages ?? 1;

  // Fetch remaining pages sequentially to keep it simple
  for (page = 2; page <= totalPages; page += 1) {
    try {
      const next = await getV2ListLibraryAgents({ page, page_size: pageSize });
      if (next.status === 200) {
        all.push(
          ...next.data.agents.map(
            (a): AgentInfo => ({
              name:
                a.name ||
                (a.graph_id ? `Agent ${a.graph_id.slice(0, 8)}` : "Agent"),
              description: a.description || "",
              library_agent_id: a.id,
              graph_id: a.graph_id,
            }),
          ),
        );
      }
    } catch (err) {
      Sentry.captureException(err, {
        tags: { context: "library_agents_fetch" },
      });
      // continue to attempt other pages
    }
  }

  return all;
}

function persistCache(cached: CachedAgents) {
  try {
    storage.set(Key.LIBRARY_AGENTS_CACHE, JSON.stringify(cached));
  } catch (error) {
    // Ignore cache failures
    console.error("Failed to persist library agents cache", error);

    Sentry.captureException(error, {
      tags: { context: "library_agents_cache_persist" },
    });
  }
}

function readCache(): CachedAgents | undefined {
  try {
    const raw = storage.get(Key.LIBRARY_AGENTS_CACHE);
    if (!raw) return;
    return JSON.parse(raw) as CachedAgents;
  } catch {
    return;
  }
}

export const useAgentStore = create<AgentStore>((set, get) => ({
  agents: [],
  lastUpdatedAt: undefined,
  isRefreshing: false,
  error: undefined,
  loadFromCache: () => {
    const cached = readCache();
    if (cached?.agents?.length) {
      set({ agents: cached.agents, lastUpdatedAt: cached.lastUpdatedAt });
    }
  },
  refreshAll: async () => {
    if (get().isRefreshing) return;
    set({ isRefreshing: true, error: undefined });
    try {
      const agents = await fetchAllLibraryAgents();
      const snapshot: CachedAgents = { agents, lastUpdatedAt: Date.now() };
      persistCache(snapshot);
      set({ agents, lastUpdatedAt: snapshot.lastUpdatedAt });
    } catch (error) {
      set({ error });
    } finally {
      set({ isRefreshing: false });
    }
  },
}));

export function buildAgentInfoMap(agents: AgentInfo[]) {
  const map = new Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >();
  agents.forEach((a) => {
    if (a.graph_id && a.library_agent_id) {
      map.set(a.graph_id, {
        name: a.name,
        description: a.description,
        library_agent_id: a.library_agent_id,
      });
    }
  });
  return map;
}
