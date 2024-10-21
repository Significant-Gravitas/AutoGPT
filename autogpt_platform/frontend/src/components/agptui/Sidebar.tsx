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
                    <button aria-label="Open sidebar menu" className="md:hidden fixed top-1/2 left-0 border border-neutral-500 bg-neutral-200 rounded-r-xl p-1">
                        <Menu className="h-6 w-6" />
                        <span className="sr-only">Open sidebar menu</span>
                    </button>
                </SheetTrigger>
                <SheetContent side="left" className="w-[280px] sm:w-[280px] p-0">
                    <div className="h-full bg-neutral-100">
                        <div className="p-6 flex flex-col justify-start items-start gap-[30px]">
                            <h2 className="text-neutral-900 text-xl font-medium font-neue leading-7 tracking-tight">
                                Creator Dashboard
                            </h2>
                            <Separator className="self-stretch" />
                            {linkGroups.map((group, groupIndex) => (
                                <React.Fragment key={groupIndex}>
                                    {group.links.map((link, linkIndex) => (
                                        <Link
                                            key={linkIndex}
                                            href={link.href}
                                            className="self-stretch text-neutral-500 text-xl font-normal font-neue leading-7 tracking-tight hover:text-neutral-700"
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
            <div className="hidden md:block w-[280px] h-[934px] relative">
                <div className="w-full h-full absolute left-0 top-0 bg-neutral-100" />
                <div className="w-[210px] absolute left-10 top-[63px] flex flex-col justify-start items-start gap-[30px]">
                    <h2 className="self-stretch text-neutral-900 text-xl font-normal font-neue leading-7 tracking-tight hover:text-neutral-700">
                        Creator Dashboard
                    </h2>
                    <Separator className="self-stretch" />
                    {linkGroups.map((group, groupIndex) => (
                        <React.Fragment key={groupIndex}>
                            {group.links.map((link, linkIndex) => (
                                <Link
                                    key={linkIndex}
                                    href={link.href}
                                    className="self-stretch text-neutral-600 text-xl font-normal font-neue leading-7 tracking-tight hover:text-neutral-700"
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
