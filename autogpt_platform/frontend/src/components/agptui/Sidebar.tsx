import * as React from "react";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu } from "lucide-react";

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
          <button
            aria-label="Open sidebar menu"
            className="fixed left-0 top-1/2 rounded-r-xl border border-neutral-500 bg-neutral-200 p-1 lg:hidden"
          >
            <Menu className="h-6 w-6" />
            <span className="sr-only">Open sidebar menu</span>
          </button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[280px] p-0 sm:w-[280px]">
          <div className="h-full bg-neutral-100">
            <div className="flex flex-col items-start justify-start gap-[30px] p-6">
              <h2 className="font-neue text-xl font-medium leading-7 tracking-tight text-neutral-900">
                Creator Dashboard
              </h2>
              <Separator className="self-stretch" />
              {linkGroups.map((group, groupIndex) => (
                <React.Fragment key={groupIndex}>
                  {group.links.map((link, linkIndex) => (
                    <Link
                      key={linkIndex}
                      href={link.href}
                      className="self-stretch font-neue text-xl font-normal leading-7 tracking-tight text-neutral-500 hover:text-neutral-700"
                    >
                      {link.text}
                    </Link>
                  ))}
                  {groupIndex < linkGroups.length - 1 && (
                    <Separator className="self-stretch" />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </SheetContent>
      </Sheet>
      <div className="relative hidden h-[934px] w-[280px] lg:block">
        <div className="absolute left-0 top-0 h-full w-full bg-neutral-100" />
        <div className="absolute left-10 top-[63px] flex w-[210px] flex-col items-start justify-start gap-[30px]">
          <h2 className="self-stretch font-neue text-xl font-normal leading-7 tracking-tight text-neutral-900 hover:text-neutral-700">
            Creator Dashboard
          </h2>
          <Separator className="self-stretch" />
          {linkGroups.map((group, groupIndex) => (
            <React.Fragment key={groupIndex}>
              {group.links.map((link, linkIndex) => (
                <Link
                  key={linkIndex}
                  href={link.href}
                  className="self-stretch font-neue text-xl font-normal leading-7 tracking-tight text-neutral-600 hover:text-neutral-700"
                >
                  {link.text}
                </Link>
              ))}
              {groupIndex < linkGroups.length - 1 && (
                <Separator className="self-stretch" />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </>
  );
};
