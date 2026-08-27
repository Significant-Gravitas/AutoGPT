import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";

import type { Org } from "./store";

export function normalizeOrg(org: OrgResponse): Org {
  return {
    id: org.id,
    name: org.name,
    slug: org.slug,
    avatarUrl: org.avatar_url ?? null,
    isPersonal: org.is_personal,
    memberCount: org.member_count,
  };
}
