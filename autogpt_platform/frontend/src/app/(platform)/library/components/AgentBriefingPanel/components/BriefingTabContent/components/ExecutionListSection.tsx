"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import { SitrepItem } from "@/app/(platform)/library/components/SitrepItem/SitrepItem";
import type { AgentStatusFilter } from "@/app/(platform)/library/types";
import { MAX_VISIBLE } from "../helpers";
import { EmptyMessage } from "./EmptyMessage";

interface Props {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}

export function ExecutionListSection({ activeTab, agents }: Props) {
  const allItems = useSitrepItems(agents, agents.length);
  const [showAll, setShowAll] = useState(false);

  const filtered = allItems.filter((item) => {
    if (activeTab === "running") return item.priority === "running";
    if (activeTab === "attention") return item.priority === "error";
    if (activeTab === "completed") return item.priority === "success";
    return false;
  });

  if (filtered.length === 0) return <EmptyMessage tab={activeTab} />;

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  function handleToggleShowAll() {
    setShowAll(!showAll);
  }

  return (
    <div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {visible.map((item) => (
          <SitrepItem key={item.id} item={item} />
        ))}
      </div>
      {hasMore ? (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={handleToggleShowAll}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
