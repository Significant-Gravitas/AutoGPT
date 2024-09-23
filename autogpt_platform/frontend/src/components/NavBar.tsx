import Link from "next/link";
import { Button } from "@/components/ui/button";
import React from "react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import Image from "next/image";
import getServerUser from "@/hooks/getServerUser";
import ProfileDropdown from "./ProfileDropdown";
import {
  IconCircleUser,
  IconMenu,
  IconPackage2,
  IconRefresh,
  IconSquareActivity,
  IconWorkFlow,
} from "@/components/ui/icons";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import CreditButton from "@/components/CreditButton";

export async function NavBar() {
  const isAvailable = Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL &&
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
  );
  const { user } = await getServerUser();

  return (
    <header className="sticky top-0 z-50 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
      <div className="flex flex-1 items-center gap-4">
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="shrink-0 md:hidden"
            >
              <IconMenu />
              <span className="sr-only">Toggle navigation menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left">
            <nav className="grid gap-6 text-lg font-medium">
              <Link
                href="/"
                className="flex flex-row gap-2 text-muted-foreground hover:text-foreground"
              >
                <IconSquareActivity /> Monitor
              </Link>
              <Link
                href="/build"
                className="flex flex-row gap-2 text-muted-foreground hover:text-foreground"
              >
                <IconWorkFlow /> Build
              </Link>
              <Link
                href="/marketplace"
                className="flex flex-row gap-2 text-muted-foreground hover:text-foreground"
              >
                <IconPackage2 /> Marketplace
              </Link>
            </nav>
          </SheetContent>
        </Sheet>
        <nav className="hidden md:flex md:flex-row md:items-center md:gap-5 lg:gap-6">
          <Link
            href="/"
            className="flex flex-row items-center gap-2 text-muted-foreground hover:text-foreground"
          >
            <IconSquareActivity /> Monitor
          </Link>
          <Link
            href="/build"
            className="flex flex-row items-center gap-2 text-muted-foreground hover:text-foreground"
          >
            <IconWorkFlow /> Build
          </Link>
          <Link
            href="/marketplace"
            className="flex flex-row items-center gap-2 text-muted-foreground hover:text-foreground"
          >
            <IconPackage2 /> Marketplace
          </Link>
        </nav>
      </div>
      <div className="relative flex flex-1 justify-center">
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
      <div className="flex flex-1 items-center justify-end gap-4">
        {isAvailable && user && <CreditButton />}

        {isAvailable && !user && (
          <Link
            href="/login"
            className="flex flex-row items-center gap-2 text-muted-foreground hover:text-foreground"
          >
            Log In
            <IconCircleUser />
          </Link>
        )}
        {isAvailable && user && <ProfileDropdown />}
      </div>
    </header>
  );
}
