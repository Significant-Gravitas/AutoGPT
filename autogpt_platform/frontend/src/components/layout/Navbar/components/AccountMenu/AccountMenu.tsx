import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { environment } from "@/services/environment";
import * as React from "react";
import { MenuItemGroup } from "../../helpers";
import { AccountLogoutOption } from "./components/AccountLogoutOption";
import { AccountMenuActivityRow } from "./components/AccountMenuActivityRow";
import { AccountMenuFeedbackRow } from "./components/AccountMenuFeedbackRow";
import { AccountMenuHeader } from "./components/AccountMenuHeader";
import { AccountMenuRow } from "./components/AccountMenuRow";
import { AccountMenuTourRow } from "./components/AccountMenuTourRow";
import { InitialAvatar } from "./components/InitialAvatar";
import { getAccountMenuIcon } from "./helpers";

interface Props {
  userName?: string;
  userHandle?: string;
  userEmail?: string;
  avatarSrc?: string;
  menuItemGroups: MenuItemGroup[];
  isLoading?: boolean;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
}

export function AccountMenu({
  userName,
  userHandle,
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
          // TODO(tokens): no exact role for focus-visible:ring-neutral-300 (#D4D4D4)
          className="flex cursor-pointer items-center space-x-3 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-neutral-300"
          aria-label="Open profile menu"
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="profile-popout-menu-trigger"
        >
          <InitialAvatar
            src={avatarSrc}
            name={userName}
            username={userHandle}
            className="h-8 w-8"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        side={side}
        align={align}
        sideOffset={8}
        // TODO(tokens): no exact role for border-neutral-200 (#E5E5E5)
        className="w-72 overflow-hidden rounded-2xl border border-neutral-200 bg-surface px-0 py-2 shadow-lg"
        data-testid="account-menu-popover"
      >
        <div className="px-2">
          <AccountMenuHeader
            userName={userName}
            userHandle={userHandle}
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
              {showDivider && (
                <div
                  // TODO(tokens): no exact role for bg-neutral-200 (#E5E5E5)
                  className="mx-3 my-1 h-px bg-neutral-200"
                />
              )}
              <div className="px-2 py-1">
                <ul className="flex flex-col gap-0.5">
                  {/* Help lives in the last group: replaying the first-run
                      walkthrough sits with "What's new" and the docs. */}
                  {isLogoutGroup && (
                    <li>
                      <AccountMenuTourRow />
                    </li>
                  )}
                  {group.items.map((item, itemIndex) => {
                    const key = `${groupIndex}-${itemIndex}-${item.text}`;
                    const icon = getAccountMenuIcon(item.icon);

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
                  })}
                  {groupIndex === 0 && (
                    <>
                      <li>
                        <AccountMenuActivityRow />
                      </li>
                      {environment.isCloud() && (
                        <li>
                          <AccountMenuFeedbackRow />
                        </li>
                      )}
                    </>
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
