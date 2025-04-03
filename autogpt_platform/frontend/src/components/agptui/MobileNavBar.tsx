"use client";

import * as React from "react";
import Link from "next/link";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverPortal,
} from "@/components/ui/popover";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import {
  IconType,
  IconMenu,
  IconChevronUp,
  IconEdit,
  IconLayoutDashboard,
  IconUploadCloud,
  IconSettings,
  IconLogOut,
  IconMarketplace,
  IconLibrary,
  IconBuilder,
} from "../ui/icons";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { usePathname } from "next/navigation";

interface MobileNavBarProps {
  userName?: string;
  userEmail?: string;
  avatarSrc?: string;
  menuItemGroups: {
    groupName?: string;
    items: {
      icon: IconType;
      text: string;
      href?: string;
      onClick?: () => void;
    }[];
  }[];
}

const Overlay = React.forwardRef<HTMLDivElement, { children: React.ReactNode }>(
  ({ children }, ref) => (
    <div ref={ref} className="h-screen w-screen backdrop-blur-md">
      {children}
    </div>
  ),
);
Overlay.displayName = "Overlay";

const PopoutMenuItem: React.FC<{
  icon: IconType;
  isActive: boolean;
  text: React.ReactNode;
  href?: string;
  onClick?: () => void;
}> = ({ icon, isActive, text, href, onClick }) => {
  const getIcon = (iconType: IconType) => {
    const iconClass = "w-6 h-6 relative";
    switch (iconType) {
      case IconType.Marketplace:
        return <IconMarketplace className={iconClass} />;
      case IconType.Library:
        return <IconLibrary className={iconClass} />;
      case IconType.Builder:
        return <IconBuilder className={iconClass} />;
      case IconType.Edit:
        return <IconEdit className={iconClass} />;
      case IconType.LayoutDashboard:
        return <IconLayoutDashboard className={iconClass} />;
      case IconType.UploadCloud:
        return <IconUploadCloud className={iconClass} />;
      case IconType.Settings:
        return <IconSettings className={iconClass} />;
      case IconType.LogOut:
        return <IconLogOut className={iconClass} />;
      default:
        return null;
    }
  };

  const content = (
    <div className="inline-flex w-full items-center justify-start gap-4 hover:rounded hover:bg-[#e0e0e0] dark:hover:bg-[#3a3a3a]">
      {getIcon(icon)}
      <div className="relative">
        <div
          className={`font-inter text-base font-normal leading-7 text-[#474747] dark:text-[#cfcfcf] ${isActive ? "font-semibold text-[#272727] dark:text-[#ffffff]" : "text-[#474747] dark:text-[#cfcfcf]"}`}
        >
          {text}
        </div>
        {isActive && (
          <div className="absolute bottom-[-4px] left-0 h-[2px] w-full bg-[#272727] dark:bg-[#ffffff]"></div>
        )}
      </div>
    </div>
  );

  if (onClick)
    return (
      <div className="w-full" onClick={onClick}>
        {content}
      </div>
    );
  if (href)
    return (
      <Link href={href} className="w-full">
        {content}
      </Link>
    );
  return content;
};

export const MobileNavBar: React.FC<MobileNavBarProps> = ({
  userName,
  userEmail,
  avatarSrc,
  menuItemGroups,
}) => {
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
                    <div className="absolute left-0 top-6 font-inter text-base font-normal leading-7 text-[#474747] dark:text-[#cfcfcf]">
                      {userEmail || "No Email Set"}
                    </div>
                  </div>
                </div>
                <Separator className="mb-4 dark:bg-[#3a3a3a]" />
                {menuItemGroups.map((group, groupIndex) => (
                  <React.Fragment key={groupIndex}>
                    {group.items.map((item, itemIndex) => (
                      <PopoutMenuItem
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
};
