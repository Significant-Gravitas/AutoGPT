"use client";

import { Text } from "@/components/atoms/Text/Text";
import { FolderIcon, type Icon } from "@phosphor-icons/react";
import { usePathname } from "next/navigation";

const ROUTE_TITLES: Record<string, { title: string; icon: Icon }> = {
  "/artifacts": { title: "Files", icon: FolderIcon },
};

function getRouteTitle(pathname: string | null) {
  if (!pathname) return null;
  const match = Object.entries(ROUTE_TITLES).find(
    ([href]) => pathname === href || pathname.startsWith(`${href}/`),
  );
  return match ? match[1] : null;
}

export function InsetHeaderTitle() {
  const pathname = usePathname();
  const entry = getRouteTitle(pathname);

  if (!entry) return null;

  const TitleIcon = entry.icon;

  return (
    <div className="flex items-center gap-2">
      <TitleIcon className="size-5 text-zinc-800" />
      <Text variant="large-medium">{entry.title}</Text>
    </div>
  );
}
