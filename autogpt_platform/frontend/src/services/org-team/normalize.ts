import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";

import type { Org } from "./store";
import { resolveOrgAvatarUrl } from "./avatar";

export function normalizeOrg(org: OrgResponse): Org {
  return {
    id: org.id,
    name: org.name,
    slug: org.slug,
    avatarUrl: resolveOrgAvatarUrl(org.avatar_url ?? null),
    isPersonal: org.is_personal,
    memberCount: org.member_count,
  };
}
