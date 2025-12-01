"use client";

import {
  Popover,
  PopoverContent,
  PopoverPortal,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Separator } from "@/components/__legacy__/ui/separator";
import { AnimatePresence, motion } from "framer-motion";
import { usePathname } from "next/navigation";
import * as React from "react";
import { MenuItemGroup } from "../../helpers";
import { MobileNavbarMenuItem } from "./components/MobileNavbarMenuItem";
import { Button } from "@/components/atoms/Button/Button";
import { CaretUpIcon, ListIcon } from "@phosphor-icons/react";
import Avatar, {
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";

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
          variant="ghost"
          aria-label="Open menu"
          className="min-w-auto flex !min-w-[3.75rem] items-center justify-center md:hidden"
          data-testid="mobile-nav-bar-trigger"
        >
          {isOpen ? (
            <CaretUpIcon className="size-6 stroke-slate-800" />
          ) : (
            <ListIcon className="size-6 stroke-slate-800" />
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
                className="w-screen rounded-b-2xl bg-white"
              >
                <div className="mb-4 inline-flex w-full items-end justify-start gap-4">
                  <Avatar className="h-14 w-14">
                    <AvatarImage
                      src={avatarSrc}
                      alt={userName || "Unknown User"}
                    />
                    <AvatarFallback>
                      {userName?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="relative h-14 w-full">
                    <div className="absolute left-0 top-0 text-lg font-semibold leading-7 text-[#474747]">
                      {userName || "Unknown User"}
                    </div>
                    <div className="absolute left-0 top-6 font-sans text-base font-normal leading-7 text-[#474747]">
                      {userEmail || "No Email Set"}
                    </div>
                  </div>
                </div>
                <Separator className="mb-4" />
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
                      <Separator className="my-4" />
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
