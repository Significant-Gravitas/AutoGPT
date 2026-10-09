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

  return (
    <>
      <motion.div
        variants={itemVariants}
        className="hidden group-data-[collapsible=icon]:block"
      >
        <ChatsRailItem onClick={jumpToRecentChats} />
      </motion.div>

      <motion.div
        ref={recentChatsRef}
        variants={itemVariants}
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
      </motion.div>
    </>
  );
}
