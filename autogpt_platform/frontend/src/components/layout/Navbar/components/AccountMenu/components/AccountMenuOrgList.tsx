"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { CheckIcon, GearSixIcon, PlusIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { useOrgTeamSwitcher } from "../../OrgTeamSwitcher/useOrgTeamSwitcher";

export function AccountMenuOrgList() {
  const { orgs, teams, activeOrg, activeTeam, switchOrg, switchTeam } =
    useOrgTeamSwitcher();

  // UI-only for now. To wire this up: call the generated
  // `usePostV2CreateOrganization` hook (POST /api/orgs) with a
  // `CreateOrgRequest` ({ name, slug, description? }); on success invalidate
  // the org list query so `OrgTeamProvider` refetches GET /api/orgs, then
  // `switchOrg(newOrg.id)` to make the new org active.
  function handleCreateOrganization() {
    // TODO: connect to backend (usePostV2CreateOrganization) + revalidate orgs.
  }

  const createOrgButton = (
    <button
      type="button"
      className="flex w-full items-center gap-2 rounded-lg bg-neutral-100 px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-200"
      onClick={handleCreateOrganization}
      data-testid="create-organization-button"
    >
      <span className="flex h-5 w-5 items-center justify-center">
        <PlusIcon size={14} />
      </span>
      <span className="flex-1 truncate text-left">Create organization</span>
    </button>
  );

  if (orgs.length === 0) {
    return (
      <div className="flex flex-col gap-2 p-2">
        <div className="px-2 py-1 text-sm text-neutral-500">
          No organizations yet
        </div>
        {createOrgButton}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="flex flex-col gap-0.5">
        <span className="px-2 py-1 text-xs font-medium uppercase text-neutral-400">
          Organizations
        </span>
        {orgs.map((org) => (
          <button
            key={org.id}
            type="button"
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
            onClick={() => switchOrg(org.id)}
          >
            <Avatar className="h-5 w-5">
              <AvatarImage src={org.avatarUrl ?? ""} alt="" />
              <AvatarFallback className="text-xs">
                {org.name.charAt(0)}
              </AvatarFallback>
            </Avatar>
            <span className="flex-1 truncate text-left">{org.name}</span>
            {org.isPersonal && (
              <span className="text-xs text-neutral-400">Personal</span>
            )}
            {org.id === activeOrg?.id && (
              <CheckIcon size={14} className="text-green-600" />
            )}
          </button>
        ))}
        {createOrgButton}
      </div>

      {teams.length > 0 && (
        <>
          <div className="border-t border-neutral-100" />
          <div className="flex flex-col gap-0.5">
            <span className="px-2 py-1 text-xs font-medium uppercase text-neutral-400">
              Teams
            </span>
            {teams.map((team) => (
              <button
                key={team.id}
                type="button"
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
                onClick={() => switchTeam(team.id)}
              >
                <span className="flex-1 truncate text-left">{team.name}</span>
                {team.joinPolicy === "PRIVATE" && (
                  <span className="text-xs text-neutral-400">Private</span>
                )}
                {team.id === activeTeam?.id && (
                  <CheckIcon size={14} className="text-green-600" />
                )}
              </button>
            ))}
            <Link
              href="/org/teams"
              className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
            >
              <GearSixIcon size={14} />
              <span>Manage teams</span>
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
