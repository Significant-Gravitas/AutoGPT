"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Separator } from "@/components/atoms/Separator/Separator";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { usePathname } from "next/navigation";
import * as React from "react";
import { MenuItemGroup } from "../../helpers";
import { MobileNavbarLogoutItem } from "./components/MobileNavbarLogoutItem";
import { MobileNavbarMenuItem } from "./components/MobileNavbarMenuItem";
import { ArrowUp01Icon, Menu01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface MobileNavBarProps {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  menuItemGroups: MenuItemGroup[];
}

export function MobileNavBar({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
}: MobileNavBarProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const pathname = usePathname();
  const parts = pathname.split("/");
  const activeLink = parts.length > 1 ? parts[1] : parts[0];

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger
        render={
          <Button
            variant="ghost"
            aria-label="Open menu"
            className="flex min-w-15 items-center justify-center md:hidden"
            data-testid="mobile-nav-bar-trigger"
          >
            {isOpen ? (
              <Icon icon={ArrowUp01Icon} className="size-6 stroke-slate-800" />
            ) : (
              <Icon icon={Menu01Icon} className="size-6 stroke-slate-800" />
            )}
            <span className="sr-only">Open menu</span>
          </Button>
        }
      />
      <PopoverContent
        sideOffset={0}
        className="w-screen rounded-t-none rounded-b-2xl p-4 [&_[data-slot=popover-arrow]]:hidden"
      >
        <div className="mb-4 inline-flex w-full items-end justify-start gap-4">
          <Avatar className="h-14 w-14">
            <AvatarImage src={avatarSrc} alt={userName || "Unknown User"} />
            <AvatarFallback>{userName}</AvatarFallback>
          </Avatar>
          <div className="relative h-14 w-full">
            <div className="absolute top-0 left-0 text-lg leading-7 font-semibold text-zinc-800">
              {userName || "Unknown User"}
            </div>
            <div className="absolute top-6 left-0 font-sans text-base leading-7 font-normal text-zinc-800">
              {userEmail || "No Email Set"}
            </div>
          </div>
        </div>
        <Separator className="mb-4" />
        {menuItemGroups.map((group, groupIndex) => (
          <React.Fragment key={groupIndex}>
            {group.items.map((item, itemIndex) => {
              if (item.text === "Log out") {
                return (
                  <MobileNavbarLogoutItem
                    key={itemIndex}
                    icon={item.icon}
                    text={item.text}
                  />
                );
              }
              return (
                <MobileNavbarMenuItem
                  key={itemIndex}
                  icon={item.icon}
                  isActive={item.href === activeLink}
                  text={item.text}
                  onClick={item.onClick}
                  href={item.href}
                />
              );
            })}
            {groupIndex < menuItemGroups.length - 1 && (
              <Separator className="my-4" />
            )}
          </React.Fragment>
        ))}
      </PopoverContent>
    </Popover>
  );
}
