import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import Link from "next/link";
import * as React from "react";
import { getAccountMenuOptionIcon, MenuItemGroup } from "../../helpers";
import { AccountLogoutOption } from "./components/AccountLogoutOption";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";

interface Props {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  hideNavBarUsername?: boolean;
  menuItemGroups: MenuItemGroup[];
}

export function AccountMenu({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
}: Props) {
  const popupId = React.useId();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex cursor-pointer items-center space-x-3"
          aria-label="Open profile menu"
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="profile-popout-menu-trigger"
        >
          <Avatar>
            <AvatarImage src={avatarSrc} alt="" aria-hidden="true" />
            <AvatarFallback aria-hidden="true">
              {userName?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        className="flex flex-col items-start justify-start gap-4 rounded-[26px] bg-zinc-400/70 p-4 shadow backdrop-blur-2xl dark:bg-zinc-800/70"
        data-testid="account-menu-popover"
      >
        {/* Header with avatar and user info */}
        <div className="inline-flex items-center justify-start gap-1 self-stretch">
          <Avatar className="h-[60px] w-[60px]">
            <AvatarImage src={avatarSrc} alt="" aria-hidden="true" />
            <AvatarFallback aria-hidden="true">
              {userName?.charAt(0) || "U"}
            </AvatarFallback>
          </Avatar>
          <div className="relative flex h-[47px] w-[173px] flex-col items-start justify-center gap-1">
            <div className="max-w-[10.5rem] truncate font-sans text-base font-semibold leading-none text-white dark:text-neutral-200">
              {userName}
            </div>
            <div
              data-testid="account-menu-user-email"
              className="max-w-[10.5rem] truncate font-sans text-base font-normal leading-none text-white dark:text-neutral-400"
            >
              {userEmail}
            </div>
          </div>
        </div>

        {/* Menu items */}
        <div className="flex w-full flex-col items-start justify-start gap-2 rounded-[23px]">
          {menuItemGroups.map((group, groupIndex) => (
            <div
              key={groupIndex}
              className="flex w-full flex-col items-start justify-start gap-5 rounded-[18px] bg-white p-3.5 dark:bg-neutral-900"
            >
              {group.items.map((item, itemIndex) => {
                if (item.href) {
                  return (
                    <Link
                      key={itemIndex}
                      href={item.href}
                      className="inline-flex w-full items-center justify-start gap-2.5"
                    >
                      <div className="relative h-6 w-6">
                        {getAccountMenuOptionIcon(item.icon)}
                      </div>
                      <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
                        {item.text}
                      </div>
                    </Link>
                  );
                } else if (item.text === "Log out") {
                  return <AccountLogoutOption key={itemIndex} />;
                } else if (item.text === "Publish an agent") {
                  return (
                    <PublishAgentModal
                      key={itemIndex}
                      trigger={
                        <div className="inline-flex w-full flex-row flex-nowrap items-center justify-start gap-2.5">
                          <div className="relative h-6 w-6">
                            {getAccountMenuOptionIcon(item.icon)}
                          </div>
                          <div className="font-sans text-base font-medium leading-normal text-neutral-800">
                            {item.text}
                          </div>
                        </div>
                      }
                    />
                  );
                } else {
                  return (
                    <div
                      key={itemIndex}
                      className="inline-flex w-full items-center justify-start gap-2.5"
                      onClick={item.onClick}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="relative h-6 w-6">
                        {getAccountMenuOptionIcon(item.icon)}
                      </div>
                      <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
                        {item.text}
                      </div>
                    </div>
                  );
                }
              })}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
