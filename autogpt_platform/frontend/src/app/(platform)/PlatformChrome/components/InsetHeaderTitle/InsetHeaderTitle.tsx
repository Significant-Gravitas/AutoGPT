"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { usePathname } from "next/navigation";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

const ROUTE_TITLES: Record<string, { title: string; icon: IconSvgElement }> = {
  "/artifacts": { title: "Files", icon: Folder01Icon },
};

export function getRouteTitle(pathname: string | null) {
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

  return (
    <div className="flex items-center gap-2">
      <Icon icon={entry.icon} className="size-5 text-zinc-800" />
      <Text variant="large-medium">{entry.title}</Text>
    </div>
  );
}
