import * as React from "react";
import Link from "next/link";
import { Button } from "./Button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu } from "lucide-react";
import { IconDashboardLayout, IconIntegrations, IconProfile, IconSettings } from "../ui/icons";

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
            className="fixed left-4 top-4 z-40 flex h-14 w-14 items-center justify-center rounded-lg border border-neutral-500 bg-neutral-200 md:block lg:hidden"
          >
            <Menu className="h-8 w-8" />
            <span className="sr-only">Open sidebar menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[280px] p-0 sm:w-[280px] border-none">
          <div className="h-full w-full bg-zinc-200 rounded-2xl">
            <div className="h-[264px] flex-col justify-start items-start gap-6 inline-flex p-3">
              <Link href="/dashboard" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
                <IconDashboardLayout className="w-6 h-6" />
                <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                  Agent dashboard
                </div>
              </Link>
              <Link href="/integrations" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
                <IconIntegrations className="w-6 h-6" />
                <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                  Integrations
                </div>
              </Link>
              <Link href="/profile" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
                <IconProfile className="w-6 h-6" />
                <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                  Profile
                </div>
              </Link>
              <Link href="/settings" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
                <IconSettings className="w-6 h-6" />
                <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                  Settings
                </div>
              </Link>
            </div>
          </div>
        </SheetContent>
      </Sheet>
      
      <div className="relative hidden w-[234px] h-[912px] lg:block border-none">
        <div className="h-full w-full bg-zinc-200 rounded-2xl">
          <div className="h-[264px] flex-col justify-start items-start gap-6 inline-flex p-3">
            <Link href="/dashboard" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
              <IconDashboardLayout className="w-6 h-6" />
              <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                Agent dashboard
              </div>
            </Link>
            <Link href="/integrations" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
              <IconIntegrations className="w-6 h-6" />
              <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                Integrations
              </div>
            </Link>
            <Link href="/profile" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
              <IconProfile className="w-6 h-6" />
              <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                Profile
              </div>
            </Link>
            <Link href="/settings" className="self-stretch px-7 py-3 rounded-xl justify-center items-center gap-2.5 inline-flex hover:bg-neutral-800 hover:text-white text-neutral-800">
              <IconSettings className="w-6 h-6" />
              <div className="grow shrink basis-0 text-base font-medium font-['Inter'] leading-normal">
                Settings
              </div>
            </Link>
          </div>
        </div>
      </div>
    </>
  );
};
