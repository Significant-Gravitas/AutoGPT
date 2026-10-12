"use client";

import { BubbleChatIcon } from "@hugeicons/core-free-icons";
import { motion, type Variants } from "framer-motion";
import { Suspense } from "react";
import { useJumpToNavGroup } from "../../useJumpToNavGroup";
import { CollapsibleNavGroup } from "../CollapsibleNavGroup/CollapsibleNavGroup";
import { NavRailItem } from "../NavRailItem/NavRailItem";
import { RecentChats } from "../RecentChats/RecentChats";

interface Props {
  itemVariants: Variants;
}

export function ChatsNavSection({ itemVariants }: Props) {
  const { isOpen, setIsOpen, groupRef, jumpToGroup } = useJumpToNavGroup();

  // One stagger slot: only one of the two is ever displayed.
  return (
    <motion.div variants={itemVariants}>
      <div className="hidden group-data-[collapsible=icon]:block">
        <NavRailItem
          icon={BubbleChatIcon}
          label="Chats"
          onClick={jumpToGroup}
        />
      </div>

      <div ref={groupRef} className="group-data-[collapsible=icon]:hidden">
        <CollapsibleNavGroup
          label="Recent chats"
          open={isOpen}
          onOpenChange={setIsOpen}
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
