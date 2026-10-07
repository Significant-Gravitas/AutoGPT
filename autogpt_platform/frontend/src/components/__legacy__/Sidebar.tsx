import * as React from "react";
import Link from "next/link";
import { Sheet } from "@/components/molecules/Sheet/Sheet";
import { Menu } from "lucide-react";
import { IconDashboardLayout } from "./ui/icons";

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

  // Function to render link items
  const renderLinks = () => {
    return allLinks.map((link, index) => (
      <Link
        key={`${link.href}-${index}`}
        href={link.href}
        className="inline-flex w-full items-center gap-2.5 rounded-xl px-3 py-3 text-zinc-800 hover:bg-zinc-800 hover:text-white"
      >
        {link.icon || getDefaultIconForLink()}
        <div className="p-ui-medium text-base leading-normal font-medium">
          {link.text}
        </div>
      </Link>
    ));
  };

  return (
    <>
      <Sheet
        title="Menu"
        hideTitle
        side="left"
        className="w-[280px] border-none sm:w-[280px]"
        bodyClassName="px-0 pb-0"
        trigger={
          <button
            aria-label="Open sidebar menu"
            className="fixed top-4 left-4 z-50 flex h-14 w-14 items-center justify-center overflow-hidden rounded-lg border border-zinc-500 bg-zinc-200 px-4 py-2 font-sans text-sm font-medium tracking-tight whitespace-nowrap text-zinc-800 transition-colors hover:bg-zinc-200/50 focus-visible:ring-1 focus-visible:ring-black focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 md:block lg:hidden"
          >
            <Menu className="h-8 w-8 stroke-black" />
            <span className="sr-only">Open sidebar menu</span>
          </button>
        }
      >
        <div className="h-full w-full rounded-2xl bg-zinc-200">
          <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
            {renderLinks()}
          </div>
        </div>
      </Sheet>

      <div className="relative hidden h-[912px] w-[234px] border-none lg:block">
        <div className="h-full w-full rounded-2xl bg-zinc-200">
          <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
            {renderLinks()}
          </div>
        </div>
      </div>
    </>
  );
};
