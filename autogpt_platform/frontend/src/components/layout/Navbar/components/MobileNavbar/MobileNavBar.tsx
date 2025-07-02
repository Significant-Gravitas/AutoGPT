"use client";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverPortal,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { AnimatePresence, motion } from "framer-motion";
import { usePathname } from "next/navigation";
import * as React from "react";
import { IconChevronUp, IconMenu } from "../../../../ui/icons";
import { MenuItemGroup } from "../../helpers";
import { MobileNavbarMenuItem } from "./components/MobileNavbarMenuItem";

interface MobileNavBarProps {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  menuItemGroups: MenuItemGroup[];
}

const Overlay = React.forwardRef<HTMLDivElement, { children: React.ReactNode }>(
  ({ children }, ref) => (
    <div ref={ref} className="h-screen w-screen backdrop-blur-md">
      {children}
    </div>
  ),
);

Overlay.displayName = "Overlay";

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
      <PopoverTrigger asChild>
        <Button
          aria-label="Open menu"
          className="fixed right-4 top-4 z-50 flex h-14 w-14 items-center justify-center rounded-lg border border-neutral-500 bg-neutral-200 hover:bg-gray-200/50 dark:border-neutral-700 dark:bg-neutral-800 dark:hover:bg-gray-700/50 md:hidden"
          data-testid="mobile-nav-bar-trigger"
        >
          {isOpen ? (
            <IconChevronUp className="h-8 w-8 stroke-black dark:stroke-white" />
          ) : (
            <IconMenu className="h-8 w-8 stroke-black dark:stroke-white" />
          )}
          <span className="sr-only">Open menu</span>
        </Button>
      </PopoverTrigger>
      <AnimatePresence>
        <PopoverPortal>
          <Overlay>
            <PopoverContent asChild>
              <motion.div
                initial={{ opacity: 0, y: -32 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -32, transition: { duration: 0.2 } }}
                className="w-screen rounded-b-2xl bg-white dark:bg-neutral-900"
              >
                <div className="mb-4 inline-flex w-full items-end justify-start gap-4">
                  <Avatar className="h-14 w-14 border border-[#474747] dark:border-[#cfcfcf]">
                    <AvatarImage
                      src={avatarSrc}
                      alt={userName || "Unknown User"}
                    />
                    <AvatarFallback>
                      {userName?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="relative h-14 w-full">
                    <div className="absolute left-0 top-0 text-lg font-semibold leading-7 text-[#474747] dark:text-[#cfcfcf]">
                      {userName || "Unknown User"}
                    </div>
                    <div className="absolute left-0 top-6 font-sans text-base font-normal leading-7 text-[#474747] dark:text-[#cfcfcf]">
                      {userEmail || "No Email Set"}
                    </div>
                  </div>
                </div>
                <Separator className="mb-4 dark:bg-[#3a3a3a]" />
                {menuItemGroups.map((group, groupIndex) => (
                  <React.Fragment key={groupIndex}>
                    {group.items.map((item, itemIndex) => (
                      <MobileNavbarMenuItem
                        key={itemIndex}
                        icon={item.icon}
                        isActive={item.href === activeLink}
                        text={item.text}
                        onClick={item.onClick}
                        href={item.href}
                      />
                    ))}
                    {groupIndex < menuItemGroups.length - 1 && (
                      <Separator className="my-4 dark:bg-[#3a3a3a]" />
                    )}
                  </React.Fragment>
                ))}
              </motion.div>
            </PopoverContent>
          </Overlay>
        </PopoverPortal>
      </AnimatePresence>
    </Popover>
  );
}
