import * as React from "react";
import Link from "next/link";
import { Button } from "./Button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu } from "lucide-react";
import {
  IconDashboardLayout,
  IconIntegrations,
  IconProfile,
  IconSliders,
} from "../ui/icons";

interface SidebarLinkGroup {
  links: {
    text: string;
    href: string;
  }[];
}

interface SidebarProps {
  linkGroups: SidebarLinkGroup[];
}

export const Sidebar: React.FC<SidebarProps> = ({ linkGroups }) => {
  return (
    <>
      <Sheet>
        <SheetTrigger asChild>
          <Button
            aria-label="Open sidebar menu"
            className="fixed left-4 top-4 z-50 flex h-14 w-14 items-center justify-center rounded-lg border border-neutral-500 bg-neutral-200 hover:bg-gray-200/50 md:block lg:hidden"
          >
            <Menu className="h-8 w-8 stroke-black" />
            <span className="sr-only">Open sidebar menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent
          side="left"
          className="z-50 w-[280px] border-none p-0 sm:w-[280px]"
        >
          <div className="h-full w-full rounded-2xl bg-zinc-200">
            <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
              <Link
                href="/store/dashboard"
                className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
              >
                <IconDashboardLayout className="h-6 w-6" />
                <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                  Creator dashboard
                </div>
              </Link>
              <Link
                href="/integrations"
                className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
              >
                <IconIntegrations className="h-6 w-6" />
                <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                  Integrations
                </div>
              </Link>
              <Link
                href="/store/profile"
                className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
              >
                <IconProfile className="h-6 w-6" />
                <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                  Profile
                </div>
              </Link>
              <Link
                href="/store/settings"
                className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
              >
                <IconSliders className="h-6 w-6" />
                <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                  Settings
                </div>
              </Link>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <div className="relative hidden h-[912px] w-[234px] border-none lg:block">
        <div className="h-full w-full rounded-2xl bg-zinc-200">
          <div className="inline-flex h-[264px] flex-col items-start justify-start gap-6 p-3">
            <Link
              href="/store/dashboard"
              className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
            >
              <IconDashboardLayout className="h-6 w-6" />
              <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                Agent dashboard
              </div>
            </Link>
            <Link
              href="/integrations"
              className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
            >
              <IconIntegrations className="h-6 w-6" />
              <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                Integrations
              </div>
            </Link>
            <Link
              href="/store/profile"
              className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
            >
              <IconProfile className="h-6 w-6" />
              <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                Profile
              </div>
            </Link>
            <Link
              href="/store/settings"
              className="inline-flex items-center justify-center gap-2.5 self-stretch rounded-xl px-7 py-3 text-neutral-800 hover:bg-neutral-800 hover:text-white"
            >
              <IconSliders className="h-6 w-6" />
              <div className="shrink grow basis-0 font-['Inter'] text-base font-medium leading-normal">
                Settings
              </div>
            </Link>
          </div>
        </div>
      </div>
    </>
  );
};
