import React from 'react';
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { Pencil1Icon, TimerIcon } from "@radix-ui/react-icons";

import "./globals.css";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Providers } from "@/app/providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NextGen AutoGPT",
  description: "Your one stop shop to creating AI Agents",
};

const NavBar = () => (
    <nav className="bg-white dark:bg-slate-800 p-4 flex justify-between items-center shadow">
        <div className="flex space-x-4">
            <Link href="/monitor" className={buttonVariants({ variant: "ghost" })}>
                <TimerIcon className="mr-1" /> Monitor
            </Link>
            <Link href="/build" className={buttonVariants({ variant: "ghost" })}>
                <Pencil1Icon className="mr-1" /> Build
            </Link>
        </div>
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="h-8 w-8 rounded-full">
                    <Avatar>
                        <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
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
    </nav>
);
export default function RootLayout({
 children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
      <html lang="en">
      <body className={inter.className}>
      <Providers
          attribute="class"
          defaultTheme="light"
          // Feel free to remove this line if you want to use the system theme by default
          // enableSystem
          disableTransitionOnChange
      >
        <div className="min-h-screen bg-gray-200 text-gray-900">
          <NavBar />
          <main className="mx-auto p-4">
            {children}
          </main>
        </div>
      </Providers>
      </body>
      </html>
  );
}
