import {
    DropdownMenu,
    DropdownMenuContent, DropdownMenuItem,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import Link from "next/link";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import React from "react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Pencil1Icon, TimerIcon } from "@radix-ui/react-icons";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Image from "next/image";

export function NavBar() {
    return (
        <header className="sticky top-0 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
            <div className="flex items-center gap-4 flex-1">
                <Sheet>
                    <SheetTrigger asChild>
                        <Button
                            variant="outline"
                            size="icon"
                            className="shrink-0 md:hidden"
                        >
                            <Menu className="size-5"/>
                            <span className="sr-only">Toggle navigation menu</span>
                        </Button>
                    </SheetTrigger>
                    <SheetContent side="left">
                        <nav className="grid gap-6 text-lg font-medium">
                            <Link
                                href="/monitor"
                                className="text-muted-foreground hover:text-foreground flex flex-row gap-2 "
                            >
                                <TimerIcon className="size-6" /> Monitor
                            </Link>
                            <Link
                                href="/build"
                                className="text-muted-foreground hover:text-foreground flex flex-row gap-2"
                            >
                                <Pencil1Icon className="size-6"/> Build
                            </Link>
                        </nav>
                    </SheetContent>
                </Sheet>
                <nav className="hidden md:flex md:flex-row md:items-center md:gap-5 lg:gap-6">
                    <Link
                        href="/monitor"
                        className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
                    >
                        <TimerIcon className="size-4"/> Monitor
                    </Link>
                    <Link
                        href="/build"
                        className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
                    >
                        <Pencil1Icon className="size-4"/> Build
                    </Link>
                </nav>
            </div>
            <div className="flex-1 flex justify-center relative">
                <a
                    className="pointer-events-auto flex place-items-center gap-2"
                    href="https://news.agpt.co/"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    By{" "}
                    <Image
                        src="/AUTOgpt_Logo_dark.png"
                        alt="AutoGPT Logo"
                        width={100}
                        height={20}
                        priority
                    />
                </a>
            </div>
            <div className="flex items-center gap-4 flex-1 justify-end">
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="ghost" className="size-8">
                            <Avatar>
                                <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn"/>
                                <AvatarFallback>CN</AvatarFallback>
                            </Avatar>
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        <DropdownMenuItem>Profile</DropdownMenuItem>
                        <DropdownMenuItem>Settings</DropdownMenuItem>
                        <DropdownMenuItem>Switch Workspace</DropdownMenuItem>
                        <DropdownMenuItem>Log out</DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>
        </header>
    );
}