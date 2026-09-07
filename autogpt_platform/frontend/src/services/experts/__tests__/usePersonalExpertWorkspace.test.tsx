import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { expect, test, vi } from "vitest";
import { getGetV2ListOrganizationMembersMockHandler200 } from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { usePersonalExpertWorkspace } from "../usePersonalExpertWorkspace";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-one" } }),
}));
function Wrapper({ children }: { children: ReactNode }) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

test.each([
  {
    name: "owner's personal home",
    personal: true,
    owner: true,
    team: null,
    allowed: true,
  },
  {
    name: "owner's default personal team",
    personal: true,
    owner: true,
    team: "default",
    allowed: true,
  },
  {
    name: "a nondefault team",
    personal: true,
    owner: true,
    team: "shared",
    allowed: false,
  },
  {
    name: "another owner's personal workspace",
    personal: true,
    owner: false,
    team: null,
    allowed: false,
  },
  {
    name: "a shared organization",
    personal: false,
    owner: true,
    team: null,
    allowed: false,
  },
])("limits experts for $name", async ({ personal, owner, team, allowed }) => {
  useOrgTeamStore.setState({
    activeOrgID: "org-one",
    activeTeamID: team,
    isLoaded: true,
    orgs: [
      {
        id: "org-one",
        name: "Workspace",
        slug: "workspace",
        avatarUrl: null,
        isPersonal: personal,
        memberCount: 2,
      },
    ],
    teams: [
      {
        id: "default",
        name: "Default",
        slug: "default",
        isDefault: true,
        joinPolicy: "invite_only",
        orgId: "org-one",
      },
    ],
  });
  server.use(
    getGetV2ListOrganizationMembersMockHandler200([
      {
        id: "member-one",
        user_id: "user-one",
        email: "one@example.com",
        name: "One",
        is_owner: owner,
        is_admin: false,
        is_billing_manager: false,
        joined_at: new Date(),
      },
    ]),
  );
  const { result } = renderHook(usePersonalExpertWorkspace, {
    wrapper: Wrapper,
  });
  await waitFor(() => expect(result.current.isReady).toBe(true));
  expect(result.current.canUseExperts).toBe(allowed);
});
