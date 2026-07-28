import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export interface ExpertIdentity {
  name: string;
  avatarUrl: string | null;
  role: string | null;
}

export type ExpertIdentityMap = Map<string, ExpertIdentity>;

const EMPTY_MAP: ExpertIdentityMap = new Map();

export function useExpertMap(): ExpertIdentityMap {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const expertsQuery = useListExperts({
    query: {
      enabled: isExpertsEnabled,
      select: (response) => response.data as Expert[],
    },
  });
  if (!isExpertsEnabled || !expertsQuery.data) return EMPTY_MAP;
  return new Map(
    expertsQuery.data.map((expert) => [
      expert.id,
      {
        name: expert.name,
        avatarUrl: expert.avatar_url ?? null,
        role: expert.role ?? null,
      },
    ]),
  );
}
