import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import Link from "next/link";
import { CircleUser, Menu, SquareActivity, Workflow } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import React from "react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Pencil1Icon, TimerIcon, ArchiveIcon } from "@radix-ui/react-icons";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Image from "next/image";
import getServerUser from "@/hooks/getServerUser";
import ProfileDropdown from "./ProfileDropdown";

export async function NavBar() {
  const isAvailable = Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  );
  const { user } = await getServerUser();

  return (
    <header className="sticky top-0 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6 z-50">
      <div className="flex items-center gap-4 flex-1">
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="shrink-0 md:hidden"
            >
              <Menu className="size-5" />
              <span className="sr-only">Toggle navigation menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left">
            <nav className="grid gap-6 text-lg font-medium">
              <Link
                href="/monitor"
                className="text-muted-foreground hover:text-foreground flex flex-row gap-2 "
              >
                <SquareActivity className="size-6" /> Monitor
              </Link>
              <Link
                href="/build"
                className="text-muted-foreground hover:text-foreground flex flex-row gap-2"
              >
                <Workflow className="size-6" /> Build
              </Link>
              <Link
                href="/marketplace"
                className="text-muted-foreground hover:text-foreground flex flex-row gap-2"
              >
                <ArchiveIcon className="size-6" /> Marketplace
              </Link>
            </nav>
          </SheetContent>
        </Sheet>
        <nav className="hidden md:flex md:flex-row md:items-center md:gap-5 lg:gap-6">
          <Link
            href="/monitor"
            className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
          >
            <SquareActivity className="size-4" /> Monitor
          </Link>
          <Link
            href="/build"
            className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
          >
            <Workflow className="size-4" /> Build
          </Link>
          <Link
            href="/marketplace"
            className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
          >
            <ArchiveIcon className="size-4" /> Marketplace
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
        {isAvailable && !user && (
          <Link
            href="/login"
            className="text-muted-foreground hover:text-foreground flex flex-row gap-2 items-center"
          >
            Log In
            <CircleUser className="size-5" />
          </Link>
        )}
        {isAvailable && user && <ProfileDropdown />}
      </div>
    </header>
  );
}
