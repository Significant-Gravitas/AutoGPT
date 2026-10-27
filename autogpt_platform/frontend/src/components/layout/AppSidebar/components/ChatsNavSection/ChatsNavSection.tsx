"use client";

import { motion, type Variants } from "framer-motion";
import { Suspense } from "react";
import { useJumpToRecentChats } from "../../useJumpToRecentChats";
import { ChatsRailItem } from "../ChatsRailItem/ChatsRailItem";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { RecentChats } from "../RecentChats/RecentChats";

interface Props {
  itemVariants: Variants;
}

export function ChatsNavSection({ itemVariants }: Props) {
  const {
    isRecentChatsOpen,
    setIsRecentChatsOpen,
    recentChatsRef,
    jumpToRecentChats,
  } = useJumpToRecentChats();

  // One stagger slot: only one of the two is ever displayed.
  return (
    <motion.div variants={itemVariants}>
      <div className="hidden group-data-[collapsible=icon]:block">
        <ChatsRailItem onClick={jumpToRecentChats} />
      </div>

      <div
        ref={recentChatsRef}
        className="group-data-[collapsible=icon]:hidden"
      >
        <CollapsibleNavGroup
          label="Recent chats"
          open={isRecentChatsOpen}
          onOpenChange={setIsRecentChatsOpen}
        >
          {/* Suspense boundary: RecentChats reads useSearchParams(), which
            Next.js requires to be wrapped to avoid forcing the route to
            client-side rendering. */}
          <Suspense fallback={null}>
            <RecentChats />
          </Suspense>
        </CollapsibleNavGroup>
      </div>
    </motion.div>
  );
}
