import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { WorkloadIdentityStatus } from "@/app/api/__generated__/models/workloadIdentityStatus";

export function makeIdentity(
  overrides: Partial<WorkloadIdentityInfo> = {},
): WorkloadIdentityInfo {
  return {
    id: "wi-123",
    name: "GitHub Actions · acme/api",
    issuer: "https://token.actions.githubusercontent.com",
    audience: "http://localhost:8006",
    claim_conditions: [{ claim: "repository", pattern: "acme/api" }],
    permissions: [APIKeyPermission.IDENTITY],
    token_ttl_seconds: 900,
    status: WorkloadIdentityStatus.ACTIVE,
    created_at: new Date("2026-10-22T00:00:00Z"),
    last_used_at: null,
    user_id: "user-1",
    ...overrides,
  };
}
