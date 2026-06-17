import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { useOrgTeamSwitcher } from "./useOrgTeamSwitcher";
import { CaretDown, Check, Plus, GearSix } from "@phosphor-icons/react";
import Link from "next/link";

export function OrgTeamSwitcher() {
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
          <CaretDown size={12} />
        </button>
      </PopoverTrigger>

      <PopoverContent
        className="flex w-64 flex-col gap-2 rounded-xl bg-white p-2 shadow-lg"
        align="end"
        data-testid="org-switcher-popover"
      >
        {/* Org list */}
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
                <Check size={14} className="text-green-600" />
              )}
            </button>
          ))}
          <Link
            href="/org/settings"
            className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
          >
            <Plus size={14} />
            <span>Create organization</span>
          </Link>
        </div>

        {/* Team list (only if orgs exist) */}
        {teams.length > 0 && (
          <>
            <div className="border-t border-neutral-100" />
            <div className="flex flex-col gap-0.5">
              <span className="px-2 py-1 text-xs font-medium uppercase text-neutral-400">
                Teams
              </span>
              {teams.map((ws) => (
                <button
                  key={ws.id}
                  type="button"
                  className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-neutral-100"
                  onClick={() => switchTeam(ws.id)}
                >
                  <span className="flex-1 truncate text-left">{ws.name}</span>
                  {ws.joinPolicy === "PRIVATE" && (
                    <span className="text-xs text-neutral-400">Private</span>
                  )}
                  {ws.id === activeTeam?.id && (
                    <Check size={14} className="text-green-600" />
                  )}
                </button>
              ))}
              <Link
                href="/org/teams"
                className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-neutral-500 hover:bg-neutral-100"
              >
                <GearSix size={14} />
                <span>Manage teams</span>
              </Link>
            </div>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
