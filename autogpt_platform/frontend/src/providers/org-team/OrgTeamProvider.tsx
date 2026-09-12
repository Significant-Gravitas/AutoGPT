"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { normalizeOrg } from "@/services/org-team/normalize";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef, useState } from "react";

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
 * With collaboration enabled, restores organization selection and loads its
 * teams. With it disabled or unavailable, the server resolves the caller's
 * own personal org/default team before resource screens can mount.
 *
 * On org switch: refetches the org's teams and resets the query cache.
 */
export default function OrgTeamProvider({ children }: Props) {
  const { isLoggedIn, user, isUserLoading } = useAuth();
  const { enabled: collaborationEnabled, ready: collaborationReady } =
    useFlagStatus(Flag.SHOW_ORG_SETTINGS);
  const [personalContext, setPersonalContext] = useState<{
    userId: string;
    orgId: string;
  } | null>(null);
  const [workspaceError, setWorkspaceError] = useState(false);
  const [retry, setRetry] = useState(0);
  const {
    activeOrgID,
    activeTeamID,
    teams,
    isLoaded,
    setActiveOrg,
    setActiveTeam,
    setOrgs,
    setTeams,
    setLoaded,
    clearContext,
  } = useOrgTeamStore();

  const previousScope = useRef({ orgId: activeOrgID, teamId: activeTeamID });

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
      setPersonalContext(null);
      return;
    }
    // An unresolved flag is not a disabled cohort. Keep the saved scope
    // untouched (and resource screens gated) until LD answers or times out.
    if (!collaborationReady) return;

    let cancelled = false;
    setWorkspaceError(false);
    if (!collaborationEnabled) {
      // Drop persisted shared context before resolving the caller's own
      // personal org. isPersonal alone is insufficient: a member can also
      // belong to somebody else's personal org.
      setPersonalContext(null);
      clearContext();
    }

    async function loadOrgs() {
      try {
        const res = await fetch(
          collaborationEnabled
            ? "/api/proxy/api/orgs"
            : "/api/proxy/api/orgs/default",
          {
            headers: { "Content-Type": "application/json" },
          },
        );
        if (cancelled) return;
        if (!res.ok) {
          setLoaded(true);
          if (!collaborationEnabled) setWorkspaceError(true);
          return;
        }
        const data = await res.json();
        if (cancelled) return;
        if (!collaborationEnabled) {
          const personal = normalizeOrg((data.data ?? data) as OrgResponse);
          setOrgs([personal]);
          setActiveOrg(personal.id);
          setPersonalContext({ userId: user!.id, orgId: personal.id });
          return;
        }
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
        if (cancelled) return;
        setLoaded(true);
        if (!collaborationEnabled) setWorkspaceError(true);
      }
    }

    loadOrgs();
    return () => {
      cancelled = true;
    };
  }, [
    isLoggedIn,
    user,
    isUserLoading,
    collaborationEnabled,
    collaborationReady,
    retry,
  ]);

  // A create/join callback started before the flag changed can still select
  // an org or team afterward. Reconcile it back to the server-verified
  // personal scope instead of leaving resource screens stuck behind loading.
  useEffect(() => {
    if (
      collaborationEnabled ||
      !collaborationReady ||
      isUserLoading ||
      !isLoggedIn ||
      !user ||
      personalContext?.userId !== user.id
    ) {
      return;
    }
    if (activeOrgID !== personalContext.orgId) {
      setActiveOrg(personalContext.orgId);
      return;
    }
    const defaultTeam = teams.find(
      (team) => team.orgId === personalContext.orgId && team.isDefault,
    );
    if (defaultTeam && activeTeamID !== defaultTeam.id) {
      setActiveTeam(defaultTeam.id);
    }
  }, [
    collaborationEnabled,
    collaborationReady,
    isUserLoading,
    isLoggedIn,
    user,
    personalContext,
    activeOrgID,
    activeTeamID,
    teams,
    setActiveOrg,
    setActiveTeam,
  ]);

  // Load teams for collaboration filters. Without collaboration, keep only
  // the personal default team and use it for ordinary workspace requests.
  useEffect(() => {
    if (
      !collaborationReady ||
      isUserLoading ||
      !isLoggedIn ||
      !user ||
      !activeOrgID
    ) {
      return;
    }
    if (!collaborationEnabled && personalContext?.orgId !== activeOrgID) return;

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
          if (cancelled) return;
          const teams: TeamApiShape[] = data.data || data;
          const memberTeams = teams.filter((team) => team.is_member);
          if (collaborationEnabled) {
            setTeams(memberTeams.map(mapTeam));
          } else {
            const defaultTeam = memberTeams.find((team) => team.is_default);
            if (!defaultTeam) {
              setWorkspaceError(true);
              return;
            }
            setTeams([mapTeam(defaultTeam)]);
            setActiveTeam(defaultTeam.id);
          }
        } else if (!cancelled && !collaborationEnabled) {
          setWorkspaceError(true);
        }
      } catch {
        if (!cancelled) {
          setTeams([]);
          if (!collaborationEnabled) setWorkspaceError(true);
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
  }, [
    isLoggedIn,
    user,
    isUserLoading,
    activeOrgID,
    collaborationEnabled,
    collaborationReady,
    personalContext,
  ]);

  // Drop org-scoped data when the org switches. resetQueries (NOT
  // clear) — clear() removes queries without notifying mounted
  // observers, which leaves them pending forever; resetQueries
  // refetches everything that's still on screen.
  useEffect(() => {
    if (
      previousScope.current.orgId !== null &&
      (previousScope.current.orgId !== activeOrgID ||
        previousScope.current.teamId !== activeTeamID)
    ) {
      const queryClient = getQueryClient();
      queryClient.resetQueries();
    }
    previousScope.current = { orgId: activeOrgID, teamId: activeTeamID };
  }, [activeOrgID, activeTeamID]);

  // Do not mount resource screens under a stale shared scope while the flag
  // is unavailable/off. A failed default lookup must never restore it.
  if (!collaborationReady && (isLoggedIn || isUserLoading)) {
    return (
      <div role="status" className="p-6">
        Loading your workspace…
      </div>
    );
  }
  if (!collaborationEnabled && isLoggedIn) {
    if (workspaceError) {
      return (
        <ErrorCard
          responseError={{ message: "We couldn't load your workspace." }}
          context="workspace"
          onRetry={() => setRetry((value) => value + 1)}
        />
      );
    }
    if (
      personalContext?.userId !== user?.id ||
      personalContext?.orgId !== activeOrgID ||
      !isLoaded ||
      !activeTeamID ||
      !teams.some(
        (team) =>
          team.id === activeTeamID &&
          team.isDefault &&
          team.orgId === activeOrgID,
      )
    ) {
      return (
        <div role="status" className="p-6">
          Loading your workspace…
        </div>
      );
    }
  }

  return <>{children}</>;
}
