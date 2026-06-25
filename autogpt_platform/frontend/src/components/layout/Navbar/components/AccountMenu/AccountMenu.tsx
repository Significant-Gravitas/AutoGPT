import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import * as React from "react";
import { MenuItemGroup } from "../../helpers";
import { AccountLogoutOption } from "./components/AccountLogoutOption";
import { AccountMenuRow } from "./components/AccountMenuRow";
import { InitialAvatar } from "./components/InitialAvatar";
import { getAccountMenuPhosphorIcon } from "./helpers";

interface Props {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  hideNavBarUsername?: boolean;
  menuItemGroups: MenuItemGroup[];
  isLoading?: boolean;
}

export function AccountMenu({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
  isLoading = false,
}: Props) {
  const popupId = React.useId();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex cursor-pointer items-center space-x-3 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
          aria-label="Open profile menu"
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="profile-popout-menu-trigger"
        >
          <InitialAvatar src={avatarSrc} name={userName} className="h-8 w-8" />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        align="end"
        sideOffset={8}
        className="w-64 overflow-hidden rounded-2xl border border-neutral-200 bg-white p-0 shadow-lg"
        data-testid="account-menu-popover"
      >
        <div className="flex items-center gap-3 px-4 py-3">
          <InitialAvatar src={avatarSrc} name={userName} />
          <div className="flex min-w-0 flex-1 flex-col gap-1">
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
        </div>

        <div className="border-t border-neutral-100 p-2">
          <ul className="flex flex-col gap-0.5">
            {menuItemGroups.map((group, groupIndex) =>
              group.items.map((item, itemIndex) => {
                const key = `${groupIndex}-${itemIndex}-${item.text}`;
                const icon = getAccountMenuPhosphorIcon(item.icon);

                if (item.text === "Log out") {
                  return (
                    <li key={key}>
                      <AccountLogoutOption />
                    </li>
                  );
                }

                if (item.href) {
                  return (
                    <li key={key}>
                      <AccountMenuRow
                        as="link"
                        href={item.href}
                        external={item.external}
                        icon={icon}
                        label={item.text}
                      />
                    </li>
                  );
                }

                return (
                  <li key={key}>
                    <AccountMenuRow
                      as="button"
                      onClick={item.onClick}
                      icon={icon}
                      label={item.text}
                    />
                  </li>
                );
              }),
            )}
          </ul>
        </div>
      </PopoverContent>
    </Popover>
  );
}
