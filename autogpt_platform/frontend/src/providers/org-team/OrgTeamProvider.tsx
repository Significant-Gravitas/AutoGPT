"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { normalizeOrg } from "@/services/org-team/normalize";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { useEffect, useRef } from "react";

interface Props {
  children: React.ReactNode;
}

interface TeamApiShape {
  id: string;
  name: string;
  slug: string | null;
  is_default: boolean;
  join_policy: string;
  org_id: string;
  is_member: boolean;
}

function mapTeam(team: TeamApiShape) {
  return {
    id: team.id,
    name: team.name,
    slug: team.slug,
    isDefault: team.is_default,
    joinPolicy: team.join_policy,
    orgId: team.org_id,
  };
}

/**
 * Initializes org/team context on login and clears it on logout.
 *
 * On mount (when logged in):
 * 1. Fetches the user's org list from GET /api/orgs
 * 2. If no activeOrgID is stored, sets the personal org as default
 * 3. Fetches the active org's teams so badges/filters have data
 *
 * On org switch: refetches the org's teams and resets the query cache.
 */
export default function OrgTeamProvider({ children }: Props) {
  const { isLoggedIn, user, isUserLoading } = useAuth();
  const {
    activeOrgID,
    setActiveOrg,
    setOrgs,
    setTeams,
    setLoaded,
    clearContext,
  } = useOrgTeamStore();

  const prevOrgID = useRef(activeOrgID);

  // Fetch orgs when logged in
  useEffect(() => {
    // While the session is still hydrating, isLoggedIn is transiently
    // false — clearing context here would flip activeOrgID to null and
    // (via the effect below) wipe the query cache mid-flight, stranding
    // every in-flight page query in a forever-pending state.
    if (isUserLoading) {
      return;
    }

    if (!isLoggedIn || !user) {
      clearContext();
      return;
    }

    async function loadOrgs() {
      try {
        const res = await fetch("/api/proxy/api/orgs", {
          headers: { "Content-Type": "application/json" },
        });
        if (!res.ok) {
          setLoaded(true);
          return;
        }
        const data = await res.json();
        // The API responds in snake_case; normalize to the camelCase
        // shape the store and its consumers expect.
        const rawOrgs: OrgResponse[] = data.data || data;
        const orgs = rawOrgs.map(normalizeOrg);
        setOrgs(orgs);
        if (orgs.length === 0) {
          setLoaded(true);
          return;
        }

        if (!activeOrgID || !orgs.some((org) => org.id === activeOrgID)) {
          const personal = orgs.find((o) => o.isPersonal);
          setActiveOrg(personal?.id ?? orgs[0].id);
        }
      } catch {
        setLoaded(true);
      }
    }

    loadOrgs();
  }, [isLoggedIn, user, isUserLoading]);

  // Load the active org's teams so filters/badges have data. Teams are
  // no longer a context switch, so this never selects an active team —
  // activeTeamID stays null unless a screen sets it explicitly.
  useEffect(() => {
    if (isUserLoading || !isLoggedIn || !user || !activeOrgID) {
      return;
    }

    let cancelled = false;
    setTeams([]);
    setLoaded(false);

    async function loadTeams(orgID: string) {
      try {
        const res = await fetch(`/api/proxy/api/orgs/${orgID}/workspaces`, {
          headers: {
            "Content-Type": "application/json",
            [ORG_HEADER_NAME]: orgID,
            [TEAM_HEADER_NAME]: "",
          },
        });
        if (res.ok && !cancelled) {
          const data = await res.json();
          const teams: TeamApiShape[] = data.data || data;
          setTeams(teams.filter((team) => team.is_member).map(mapTeam));
        }
      } catch {
        if (!cancelled) {
          setTeams([]);
        }
      } finally {
        if (!cancelled) {
          setLoaded(true);
        }
      }
    }

    loadTeams(activeOrgID);

    return () => {
      cancelled = true;
    };
  }, [isLoggedIn, user, isUserLoading, activeOrgID]);

  // Drop org-scoped data when the org switches. resetQueries (NOT
  // clear) — clear() removes queries without notifying mounted
  // observers, which leaves them pending forever; resetQueries
  // refetches everything that's still on screen.
  useEffect(() => {
    if (prevOrgID.current !== activeOrgID && prevOrgID.current !== null) {
      const queryClient = getQueryClient();
      queryClient.resetQueries();
    }
    prevOrgID.current = activeOrgID;
  }, [activeOrgID]);

  return <>{children}</>;
}
