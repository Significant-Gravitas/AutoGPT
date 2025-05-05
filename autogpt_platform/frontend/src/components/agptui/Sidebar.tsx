"use client";
import * as React from "react";
import Link from "next/link";
import { IconDashboardLayout } from "../ui/icons";
import { usePathname } from "next/navigation";

export interface SidebarLink {
  text: string;
  href: string;
  icon?: React.ReactNode;
}

export interface SidebarLinkGroup {
  links: SidebarLink[];
}

export interface SidebarProps {
  linkGroups: SidebarLinkGroup[];
}

// Helper function to get the default icon component based on link text
const getDefaultIconForLink = () => {
  // Default icon
  return <IconDashboardLayout className="h-6 w-6" />;
};

export const Sidebar: React.FC<SidebarProps> = ({ linkGroups }) => {
  // Extract all links from linkGroups
  const allLinks = linkGroups.flatMap((group) => group.links);
  const pathname = usePathname();

  // Function to render link items
  const renderLinks = () => {
    return allLinks.map((link, index) => {
      const isActive = pathname === link.href;
      return (
        <Link
          key={`${link.href}-${index}`}
          href={link.href}
          className={`inline-flex w-full items-center gap-2.5 rounded-xl p-3 ${
            isActive
              ? "bg-zinc-800 text-white dark:bg-neutral-700 dark:text-white"
              : "text-neutral-800 hover:bg-zinc-800 hover:text-white dark:text-neutral-200 dark:hover:bg-neutral-700 dark:hover:text-white"
          }`}
        >
          {link.icon || getDefaultIconForLink()}
          <p className="font-sans text-base font-medium">{link.text}</p>
        </Link>
      );
    });
  };

  return (
    <div className="sticky top-24 flex h-[calc(100vh-7rem)] w-60 flex-col gap-6 rounded-[1rem] bg-zinc-200 p-3">
      {renderLinks()}
    </div>
  );
};
