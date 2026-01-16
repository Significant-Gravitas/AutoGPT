"use client";

import { usePathname } from "next/navigation";
import { Children, ReactNode } from "react";

interface PlatformLayoutContentProps {
  children: ReactNode;
}

export function PlatformLayoutContent({
  children,
}: PlatformLayoutContentProps) {
  const pathname = usePathname();
  const isAuthPage =
    pathname?.includes("/login") || pathname?.includes("/signup");

  // Extract Navbar, AdminImpersonationBanner, and page content from children
  const childrenArray = Children.toArray(children);
  const navbar = childrenArray[0];
  const adminBanner = childrenArray[1];
  const pageContent = childrenArray.slice(2);

  // For login/signup pages, use a simpler layout that doesn't interfere with centering
  if (isAuthPage) {
    return (
      <main className="flex min-h-screen w-full flex-col">
        {navbar}
        {adminBanner}
        <section className="flex-1">{pageContent}</section>
      </main>
    );
  }

  // For logged-in pages, use the standard layout
  return (
    <main className="flex h-screen w-full flex-col overflow-hidden">
      {navbar}
      {adminBanner}
      <section className="flex min-h-0 flex-1 overflow-auto">
        {pageContent}
      </section>
    </main>
  );
}
