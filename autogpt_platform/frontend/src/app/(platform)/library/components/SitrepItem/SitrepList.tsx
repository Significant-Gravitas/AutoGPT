"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ClockCounterClockwise } from "@phosphor-icons/react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useSitrepItems } from "./useSitrepItems";
import { SitrepItem } from "./SitrepItem";

interface Props {
  agents: LibraryAgent[];
  maxItems?: number;
}

export function SitrepList({ agents, maxItems = 10 }: Props) {
  const items = useSitrepItems(agents, maxItems);

  if (items.length === 0) return null;

  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5">
        <ClockCounterClockwise size={16} className="text-zinc-700" />
        <Text variant="body-medium" className="text-zinc-700">
          Recent tasks
        </Text>
      </div>
      <div className="grid grid-cols-1 gap-1 lg:grid-cols-2">
        {items.map((item) => (
          <SitrepItem key={item.id} item={item} />
        ))}
      </div>
    </div>
  );
}
