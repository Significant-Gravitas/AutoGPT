"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import { useToast } from "@/components/molecules/Toast/use-toast";

// Hand-rolled types matching backend/api/features/admin/memory_admin_routes.py.
// Once #6 merges and `pnpm generate:api` runs, these become generated; for now
// hand-rolled keeps the page testable without regenerating client code.
export interface MemoryOverview {
  user_id: string;
  group_id: string;
  entities: number;
  episodes: number;
  relates_to_edges: number;
  mentions_edges: number;
  communities: number;
}

export interface EntitySummary {
  uuid: string;
  name: string;
  summary: string | null;
}

export interface FactSummary {
  uuid: string;
  source: string;
  target: string;
  name: string | null;
  fact: string | null;
  status: string | null;
  scope: string | null;
  confidence: number | null;
  created_at: string | null;
  expired_at: string | null;
}

export interface CommunitySummary {
  uuid: string;
  name: string | null;
  summary: string | null;
  member_count: number;
}

export interface RebuildResult {
  user_id: string;
  started_at: string | null;
  communities_built: unknown;
  elapsed_seconds: number | null;
  error: string | null;
  skipped: boolean;
  skip_reason: string | null;
  activity: Record<string, unknown> | null;
  forced: boolean;
}

// Backend mounts the router with prefix="/api"; the router itself adds
// prefix="/admin/memory"; final path is /api/admin/memory/...
// Matches the convention seen in src/app/api/__generated__/endpoints/admin/admin.ts.
const BASE = "/api/admin/memory/me";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await customMutator<{ data: T; status: number; headers: Headers }>(
    path,
    { method: "GET" },
  );
  return res.data;
}

export function useMemoryVisualizer() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<
    "any" | "active" | "superseded" | "contradicted"
  >("any");
  const [force, setForce] = useState(false);

  const overview = useQuery({
    queryKey: ["admin-memory", "overview"],
    queryFn: () => fetchJson<MemoryOverview>(`${BASE}/overview`),
  });

  const entities = useQuery({
    queryKey: ["admin-memory", "entities"],
    queryFn: () =>
      fetchJson<{ user_id: string; items: EntitySummary[] }>(
        `${BASE}/entities?limit=100`,
      ),
  });

  const facts = useQuery({
    queryKey: ["admin-memory", "facts", statusFilter],
    queryFn: () =>
      fetchJson<{ user_id: string; items: FactSummary[] }>(
        `${BASE}/facts?limit=100&status=${statusFilter}`,
      ),
  });

  const communities = useQuery({
    queryKey: ["admin-memory", "communities"],
    queryFn: () =>
      fetchJson<{ user_id: string; items: CommunitySummary[] }>(
        `${BASE}/communities?limit=50`,
      ),
  });

  const rebuild = useMutation({
    mutationFn: async () => {
      const res = await customMutator<{
        data: RebuildResult;
        status: number;
        headers: Headers;
      }>(`${BASE}/communities/rebuild?force=${force}`, { method: "POST" });
      return res.data;
    },
    onSuccess: (result) => {
      if (result.skipped) {
        toast({
          title: "Rebuild skipped",
          description: `${result.skip_reason ?? "no_reason"} — ${
            result.elapsed_seconds?.toFixed(2) ?? "?"
          }s`,
        });
      } else if (result.error) {
        toast({
          title: "Rebuild failed",
          description: result.error,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Rebuild complete",
          description: `${
            result.elapsed_seconds?.toFixed(1) ?? "?"
          }s — ${JSON.stringify(result.communities_built)}`,
        });
      }
      queryClient.invalidateQueries({ queryKey: ["admin-memory"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Rebuild failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  return {
    overview,
    entities,
    facts,
    communities,
    rebuild,
    statusFilter,
    setStatusFilter,
    force,
    setForce,
  };
}
