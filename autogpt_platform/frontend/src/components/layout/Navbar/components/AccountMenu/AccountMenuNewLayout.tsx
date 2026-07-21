import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import * as React from "react";
import { MenuItemGroup } from "../../helpers";
import { AccountLogoutOption } from "./components/AccountLogoutOption";
import { AccountMenuActivityRow } from "./components/AccountMenuActivityRow";
import { AccountMenuHeader } from "./components/AccountMenuHeader";
import { AccountMenuRow } from "./components/AccountMenuRow";
import { InitialAvatar } from "./components/InitialAvatar";
import { getAccountMenuPhosphorIcon } from "./helpers";

interface Props {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  menuItemGroups: MenuItemGroup[];
  isLoading?: boolean;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
}

export function AccountMenuNewLayout({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
  isLoading = false,
  side = "top",
  align = "start",
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
        side={side}
        align={align}
        sideOffset={8}
        className="w-72 overflow-hidden rounded-2xl border border-neutral-200 bg-white px-0 py-2 shadow-lg"
        data-testid="account-menu-popover"
      >
        <div className="px-2">
          <AccountMenuHeader
            userName={userName}
            userEmail={userEmail}
            avatarSrc={avatarSrc}
            isLoading={isLoading}
          />
        </div>

        {menuItemGroups.map((group, groupIndex) => {
          const isLogoutGroup = group.items.some(
            (item) => item.text === "Log out",
          );
          const showDivider = groupIndex === 0 || isLogoutGroup;

          return (
            <React.Fragment key={`group-${groupIndex}`}>
              {showDivider && <div className="mx-3 my-1 h-px bg-neutral-200" />}
              <div className="px-2 py-1">
                <ul className="flex flex-col gap-0.5">
                  {group.items.map((item, itemIndex) => {
                    const key = `${groupIndex}-${itemIndex}-${item.text}`;
                    const icon = getAccountMenuPhosphorIcon(
                      item.icon,
                      "regular",
                    );

                    if (item.text === "Log out") {
                      return (
                        <li key={key}>
                          <AccountLogoutOption weight="regular" />
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
                            newLayout
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
                          newLayout
                        />
                      </li>
                    );
                  })}
                  {groupIndex === 0 && (
                    <li>
                      <AccountMenuActivityRow />
                    </li>
                  )}
                </ul>
              </div>
            </React.Fragment>
          );
        })}
      </PopoverContent>
    </Popover>
  );
}
