"use client";

import { StorageUsage } from "@/app/(platform)/artifacts/components/StorageUsage/StorageUsage";
import { usePathname } from "next/navigation";
import type { ComponentType } from "react";

// Right-hand companions to the inset header title, keyed the same way as
// ROUTE_TITLES so a route's title and its header widget stay together.
const ROUTE_ACTIONS: Record<string, ComponentType> = {
  "/artifacts": StorageUsage,
};

export function InsetHeaderActions() {
  const pathname = usePathname();
  const match = pathname
    ? Object.entries(ROUTE_ACTIONS).find(
        ([href]) => pathname === href || pathname.startsWith(`${href}/`),
      )
    : undefined;
  if (!match) return null;
  const Actions = match[1];

  return (
    <div className="pointer-events-auto ml-auto hidden items-center md:flex">
      <Actions />
    </div>
  );
}
