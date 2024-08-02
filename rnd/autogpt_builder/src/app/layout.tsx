import React from 'react';
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import { Providers } from "@/app/providers";
import { CircleUser, SquareActivity, Workflow } from 'lucide-react';
import getServerUser from '@/hooks/getServerUser';
import ProfileDropdown from '@/components/ProfileDropdown';

import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NextGen AutoGPT",
  description: "Your one stop shop to creating AI Agents",
};

const NavBar = async () => {
  const isAvailable = Boolean(
    process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  );
  const { user } = await getServerUser();

  return (
    <nav className="bg-white dark:bg-slate-800 p-4 flex justify-between items-center shadow">
      <div className="flex space-x-4">
        <Link href="/monitor" className={buttonVariants({ variant: "ghost" })}>
          <SquareActivity className="mr-1" /> Monitor
        </Link>
        <Link href="/build" className={buttonVariants({ variant: "ghost" })}>
          <Workflow className="mr-1" /> Build
        </Link>
      </div>
      {isAvailable && !user &&
        <Link href="/login" className={buttonVariants({ variant: "ghost" })}>
          Log In<CircleUser className="ml-1" /> 
        </Link>}
      {isAvailable && user && <ProfileDropdown/>}
    </nav>
  );
}

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
