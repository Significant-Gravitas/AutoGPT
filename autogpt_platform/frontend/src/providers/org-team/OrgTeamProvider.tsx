"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
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
        const orgs = data.data || data;
        setOrgs(orgs);

        // If no active org, set the personal org as default
        if (!activeOrgID && orgs.length > 0) {
          const personal = orgs.find(
            (o: { isPersonal: boolean }) => o.isPersonal,
          );
          if (personal) {
            setActiveOrg(personal.id);
          } else {
            setActiveOrg(orgs[0].id);
          }
        }

        setLoaded(true);
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

    async function loadTeams(orgID: string) {
      try {
        const res = await fetch(`/api/proxy/api/orgs/${orgID}/workspaces`, {
          headers: { "Content-Type": "application/json" },
        });
        if (!res.ok) {
          return;
        }
        const data = await res.json();
        const teams: TeamApiShape[] = data.data || data;
        if (!cancelled) {
          setTeams(teams.map(mapTeam));
        }
      } catch {
        // Teams are non-blocking; leave the list empty on failure.
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
