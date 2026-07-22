"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { CaretRightIcon } from "@phosphor-icons/react";
import * as React from "react";
import { AccountMenuOrgList } from "./AccountMenuOrgList";
import { InitialAvatar } from "./InitialAvatar";

interface Props {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  isLoading?: boolean;
}

export function AccountMenuHeader({
  userName,
  userEmail,
  avatarSrc,
  isLoading = false,
}: Props) {
  const popupId = React.useId();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left outline-none transition-colors data-[state=open]:bg-neutral-100 hover:bg-neutral-100 focus-visible:bg-neutral-100"
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="account-menu-org-trigger"
        >
          <InitialAvatar src={avatarSrc} name={userName} className="h-8 w-8" />
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            {isLoading || !userName || !userEmail ? (
              <>
                <Skeleton className="h-3.5 w-24" />
                <Skeleton className="h-3 w-32" />
              </>
            ) : (
              <>
                <span className="truncate text-sm font-semibold leading-tight text-neutral-900">
                  {userName}
                </span>
                <span
                  data-testid="account-menu-user-email"
                  className="truncate text-sm leading-tight text-neutral-700"
                >
                  {userEmail}
                </span>
              </>
            )}
          </div>
          <CaretRightIcon
            className="shrink-0 text-neutral-700"
            size={16}
            weight="regular"
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        side="right"
        align="start"
        sideOffset={12}
        className="w-64 rounded-2xl border border-neutral-200 p-0 shadow-lg"
        data-testid="account-menu-org-popover"
      >
        <AccountMenuOrgList />
      </PopoverContent>
    </Popover>
  );
}
