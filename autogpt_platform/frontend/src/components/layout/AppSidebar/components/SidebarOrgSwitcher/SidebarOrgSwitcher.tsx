"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { CaretUpDownIcon, CheckIcon } from "@phosphor-icons/react";

import { useOrgTeamSwitcher } from "@/components/layout/Navbar/components/OrgTeamSwitcher/useOrgTeamSwitcher";

// Sidebar-width variant of the OrgTeamSwitcher (the Navbar renders the compact
// top-bar variant). Reuses the same useOrgTeamSwitcher hook so both layouts
// share behavior. "Create organization" is intentionally omitted until the org
// management frontend ships — same as the Navbar switcher.
export function SidebarOrgSwitcher() {
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
    <SidebarFooter className="p-2">
      <SidebarMenu>
        <SidebarMenuItem>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <SidebarMenuButton
                size="lg"
                tooltip={activeOrg?.name}
                aria-label="Switch organization"
                data-testid="sidebar-org-switcher-trigger"
                className="rounded-lg data-[state=open]:bg-zinc-100 hover:bg-zinc-100"
              >
                <Avatar className="size-6 shrink-0">
                  <AvatarImage
                    src={activeOrg?.avatarUrl ?? ""}
                    alt=""
                    aria-hidden="true"
                  />
                  <AvatarFallback className="text-xs" aria-hidden="true">
                    {activeOrg?.name?.charAt(0) || "O"}
                  </AvatarFallback>
                </Avatar>
                <div className="flex min-w-0 flex-1 flex-col text-left">
                  <span className="truncate text-sm font-medium">
                    {activeOrg?.name}
                  </span>
                  {activeTeam ? (
                    <span className="truncate text-xs text-zinc-500">
                      {activeTeam.name}
                    </span>
                  ) : null}
                </div>
                <CaretUpDownIcon className="ml-auto size-4 shrink-0 text-zinc-500" />
              </SidebarMenuButton>
            </DropdownMenuTrigger>

            <DropdownMenuContent
              side="top"
              align="start"
              className="w-[--radix-popper-anchor-width] min-w-56"
              data-testid="sidebar-org-switcher-content"
            >
              <DropdownMenuLabel className="text-xs uppercase text-zinc-400">
                Organizations
              </DropdownMenuLabel>
              {orgs.map((org) => (
                <DropdownMenuItem
                  key={org.id}
                  onClick={() => switchOrg(org.id)}
                  className="gap-2"
                >
                  <Avatar className="size-5 shrink-0">
                    <AvatarImage src={org.avatarUrl ?? ""} alt="" />
                    <AvatarFallback className="text-xs">
                      {org.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                  <span className="flex-1 truncate">{org.name}</span>
                  {org.isPersonal ? (
                    <span className="text-xs text-zinc-400">Personal</span>
                  ) : null}
                  {org.id === activeOrg?.id ? (
                    <CheckIcon className="size-4 text-green-600" />
                  ) : null}
                </DropdownMenuItem>
              ))}

              {teams.length > 0 ? (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel className="text-xs uppercase text-zinc-400">
                    Teams
                  </DropdownMenuLabel>
                  {teams.map((team) => (
                    <DropdownMenuItem
                      key={team.id}
                      onClick={() => switchTeam(team.id)}
                      className="gap-2"
                    >
                      <span className="flex-1 truncate">{team.name}</span>
                      {team.joinPolicy === "PRIVATE" ? (
                        <span className="text-xs text-zinc-400">Private</span>
                      ) : null}
                      {team.id === activeTeam?.id ? (
                        <CheckIcon className="size-4 text-green-600" />
                      ) : null}
                    </DropdownMenuItem>
                  ))}
                </>
              ) : null}
            </DropdownMenuContent>
          </DropdownMenu>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarFooter>
  );
}
