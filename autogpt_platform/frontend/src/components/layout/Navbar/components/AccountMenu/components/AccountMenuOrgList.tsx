"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import Link from "next/link";
import { useOrgTeamSwitcher } from "../../OrgTeamSwitcher/useOrgTeamSwitcher";
import {
  PlusSignIcon,
  Settings02Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      // TODO(tokens): no exact role for bg-neutral-100 (#F5F5F5), text-neutral-700 (#404040), hover:bg-neutral-200 (#E5E5E5)
      className="flex w-full items-center gap-2 rounded-lg bg-neutral-100 px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-200"
      onClick={handleCreateOrganization}
      data-testid="create-organization-button"
    >
      <span className="flex h-5 w-5 items-center justify-center">
        <Icon icon={PlusSignIcon} size={14} />
      </span>
      <span className="flex-1 truncate text-left">Create organization</span>
    </button>
  );

  if (orgs.length === 0) {
    return (
      <div className="flex flex-col gap-2 p-2">
        <div
          // TODO(tokens): no exact role for text-neutral-500 (#737373)
          className="px-2 py-1 text-sm text-neutral-500"
        >
          No organizations yet
        </div>
        {createOrgButton}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="flex flex-col gap-0.5">
        <span
          // TODO(tokens): no exact role for text-neutral-400 (#A3A3A3)
          className="px-2 py-1 text-xs font-medium uppercase text-neutral-400"
        >
          Organizations
        </span>
        {orgs.map((org) => (
          <button
            key={org.id}
            type="button"
            // TODO(tokens): no exact role for text-neutral-700 (#404040), hover:bg-neutral-100 (#F5F5F5)
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
              <span
                // TODO(tokens): no exact role for text-neutral-400 (#A3A3A3)
                className="text-xs text-neutral-400"
              >
                Personal
              </span>
            )}
            {org.id === activeOrg?.id && (
              <Icon icon={Tick02Icon} size={14} className="text-success" />
            )}
          </button>
        ))}
        {createOrgButton}
      </div>

      {teams.length > 0 && (
        <>
          <div
            // TODO(tokens): no exact role for border-neutral-100 (#F5F5F5)
            className="border-t border-neutral-100"
          />
          <div className="flex flex-col gap-0.5">
            <span
              // TODO(tokens): no exact role for text-neutral-400 (#A3A3A3)
              className="px-2 py-1 text-xs font-medium uppercase text-neutral-400"
            >
              Teams
            </span>
            {teams.map((team) => (
              <button
                key={team.id}
                type="button"
                // TODO(tokens): no exact role for text-neutral-700 (#404040), hover:bg-neutral-100 (#F5F5F5)
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-100"
                onClick={() => switchTeam(team.id)}
              >
                <span className="flex-1 truncate text-left">{team.name}</span>
                {team.joinPolicy === "PRIVATE" && (
                  <span
                    // TODO(tokens): no exact role for text-neutral-400 (#A3A3A3)
                    className="text-xs text-neutral-400"
                  >
                    Private
                  </span>
                )}
                {team.id === activeTeam?.id && (
                  <Icon icon={Tick02Icon} size={14} className="text-success" />
                )}
              </button>
            ))}
            <Link
              href="/org/teams"
              // TODO(tokens): no exact role for text-neutral-500 (#737373), hover:bg-neutral-100 (#F5F5F5)
              className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
            >
              <Icon icon={Settings02Icon} size={14} />
              <span>Manage teams</span>
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
