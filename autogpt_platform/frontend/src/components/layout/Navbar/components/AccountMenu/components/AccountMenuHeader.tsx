"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import * as React from "react";
import { AccountMenuOrgList } from "./AccountMenuOrgList";
import { InitialAvatar } from "./InitialAvatar";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  userName?: string;
  userHandle?: string;
  userEmail?: string;
  avatarSrc?: string;
  isLoading?: boolean;
}

export function AccountMenuHeader({
  userName,
  userHandle,
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
          // TODO(tokens): no exact role for data-[state=open]:bg-neutral-100 (#F5F5F5), hover:bg-neutral-100 (#F5F5F5), focus-visible:bg-neutral-100 (#F5F5F5)
          className="group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left outline-none transition-colors hover:bg-neutral-100 focus-visible:bg-neutral-100 data-[state=open]:bg-neutral-100"
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="account-menu-org-trigger"
        >
          <InitialAvatar
            src={avatarSrc}
            name={userName}
            username={userHandle}
            className="h-8 w-8"
          />
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            {isLoading || !userName || !userEmail ? (
              <>
                <Skeleton className="h-3.5 w-24" />
                <Skeleton className="h-3 w-32" />
              </>
            ) : (
              <>
                <span
                  // TODO(tokens): no exact role for text-neutral-900 (#171717)
                  className="truncate text-sm font-semibold leading-tight text-neutral-900"
                >
                  {userName}
                </span>
                <span
                  data-testid="account-menu-user-email"
                  // TODO(tokens): no exact role for text-neutral-700 (#404040)
                  className="truncate text-sm leading-tight text-neutral-700"
                >
                  {userEmail}
                </span>
              </>
            )}
          </div>
          <Icon
            icon={ArrowRight01Icon}
            // TODO(tokens): no exact role for text-neutral-700 (#404040)
            className="shrink-0 text-neutral-700"
            size={16}
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        side="right"
        align="start"
        sideOffset={12}
        // TODO(tokens): no exact role for border-neutral-200 (#E5E5E5)
        className="w-64 rounded-2xl border border-neutral-200 p-0 shadow-lg"
        data-testid="account-menu-org-popover"
      >
        <AccountMenuOrgList />
      </PopoverContent>
    </Popover>
  );
}
