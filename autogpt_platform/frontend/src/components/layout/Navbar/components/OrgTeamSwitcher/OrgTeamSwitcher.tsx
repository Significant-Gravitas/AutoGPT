import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CreateOrgDialog } from "@/components/contextual/CreateOrgDialog/CreateOrgDialog";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  ArrowDown01Icon,
  PlusSignIcon,
  Settings02Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useState } from "react";
import { useOrgTeamSwitcher } from "./useOrgTeamSwitcher";

export function OrgTeamSwitcher() {
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const canManageOrgs = useGetFlag(Flag.SHOW_ORG_SETTINGS);
  const {
    orgs,
    teams,
    activeOrg,
    activeTeam,
    switchOrg,
    switchTeam,
    isLoaded,
  } = useOrgTeamSwitcher();

  if (!isLoaded || orgs.length === 0) {
    return null;
  }

  return (
    <>
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex cursor-pointer items-center gap-1.5 rounded-lg bg-white/60 px-2.5 py-1.5 text-sm font-medium text-neutral-700 hover:bg-white/80"
            aria-label="Switch organization"
            data-testid="org-switcher-trigger"
          >
            <Avatar className="h-5 w-5">
              <AvatarImage
                src={activeOrg?.avatarUrl ?? ""}
                alt=""
                aria-hidden="true"
              />
              <AvatarFallback className="text-xs" aria-hidden="true">
                {activeOrg?.name?.charAt(0) || "O"}
              </AvatarFallback>
            </Avatar>
            <span className="max-w-[8rem] truncate">{activeOrg?.name}</span>
            <Icon icon={ArrowDown01Icon} size={12} />
          </button>
        </PopoverTrigger>

        <PopoverContent
          className="flex w-64 flex-col gap-2 rounded-xl bg-white p-2 shadow-lg"
          align="end"
          data-testid="org-switcher-popover"
        >
          <div className="flex flex-col gap-0.5">
            <span className="px-2 py-1 text-xs font-medium uppercase text-neutral-400">
              Organizations
            </span>
            {orgs.map((org) => (
              <button
                key={org.id}
                type="button"
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-neutral-100"
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
                  <Icon
                    icon={Tick02Icon}
                    size={14}
                    className="text-green-600"
                  />
                )}
              </button>
            ))}
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
                    className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-neutral-100"
                    onClick={() => switchTeam(team.id)}
                  >
                    <span className="flex-1 truncate text-left">
                      {team.name}
                    </span>
                    {team.joinPolicy === "PRIVATE" && (
                      <span className="text-xs text-neutral-400">Private</span>
                    )}
                    {team.id === activeTeam?.id && (
                      <Icon
                        icon={Tick02Icon}
                        size={14}
                        className="text-green-600"
                      />
                    )}
                  </button>
                ))}
              </div>
            </>
          )}

          <div className="border-t border-neutral-100" />
          <div className="flex flex-col gap-0.5">
            <Link
              href="/settings/organization"
              className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
              data-testid="org-switcher-manage"
            >
              <Icon icon={Settings02Icon} size={14} />
              <span>Manage organization</span>
            </Link>
            {canManageOrgs && (
              <button
                type="button"
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
                onClick={() => setIsCreateOpen(true)}
                data-testid="org-switcher-create"
              >
                <Icon icon={PlusSignIcon} size={14} />
                <span>Create organization</span>
              </button>
            )}
          </div>
        </PopoverContent>
      </Popover>
      {canManageOrgs && (
        <CreateOrgDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} />
      )}
    </>
  );
}
