"use client";

import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { CreateOrgDialog } from "@/components/contextual/CreateOrgDialog/CreateOrgDialog";
import { CheckIcon, PlusIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useOrgTeamSwitcher } from "../../OrgTeamSwitcher/useOrgTeamSwitcher";

export function AccountMenuOrgList() {
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const { orgs, activeOrg, switchOrg, isLoaded } = useOrgTeamSwitcher();

  if (!isLoaded) {
    return null;
  }

  const createOrgButton = (
    <button
      type="button"
      className="flex w-full items-center gap-2 rounded-lg bg-neutral-100 px-2 py-1.5 text-sm text-neutral-700 hover:bg-neutral-200"
      onClick={() => setIsCreateOpen(true)}
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
      <>
        <div className="flex flex-col gap-2 p-2">
          <div className="px-2 py-1 text-sm text-neutral-500">
            No organizations yet
          </div>
          {createOrgButton}
        </div>
        <CreateOrgDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} />
      </>
    );
  }

  return (
    <>
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
      </div>
      <CreateOrgDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} />
    </>
  );
}
