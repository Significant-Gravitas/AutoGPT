"use client";

import { expertKickoffHref } from "@/app/(platform)/copilot/expertKickoff";
import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import { Badge } from "@/components/atoms/Badge/Badge";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import {
  SidebarMenu,
  SidebarMenuItem,
} from "@/components/molecules/Sidebar/Sidebar";
import Link, { useLinkStatus } from "next/link";
import {
  ExpertGroupIdentity,
  GroupBody,
} from "../ExpertChatGroup/ExpertChatGroup";

interface Props {
  expert: ExpertIdentity;
}

/** A new expert with no chat yet: the header their group will have, marked
 *  New, over one row that opens their first chat, where they introduce
 *  themselves. */
export function NewExpertGroup({ expert }: Props) {
  return (
    <div role="group" aria-label={expert.name}>
      {/* Spaced and sized like ExpertChatGroup's header, so nothing moves
          when the first chat turns this into that group. */}
      <div className="mb-1 flex min-w-0 items-center gap-2 px-2 py-1 text-body leading-5 font-medium max-md:py-2 max-md:text-body-lg max-md:leading-6">
        <ExpertGroupIdentity
          label={expert.name}
          role={expert.jobTitle || expert.role}
          avatarUrl={expert.avatarUrl}
          color={expert.color}
          nameAccessory={
            <Badge variant="success" size="small">
              New
            </Badge>
          }
        />
      </div>
      <GroupBody>
        <SidebarMenu>
          <SidebarMenuItem>
            <Link
              href={expertKickoffHref(expert.id)}
              className="flex h-8 w-full min-w-0 items-center gap-2 rounded-md p-2 text-foreground-base outline-none hover:bg-surface-hover hover:text-foreground-strong focus-visible:ring-2 focus-visible:ring-info max-md:h-11"
            >
              <Text
                as="span"
                variant="body-tight-touch"
                tone="inherit"
                className="truncate"
              >
                Say hello to {expert.name}
              </Text>
              <PendingSpinner />
            </Link>
          </SidebarMenuItem>
        </SidebarMenu>
      </GroupBody>
    </div>
  );
}

function PendingSpinner() {
  const { pending } = useLinkStatus();
  if (!pending) return null;
  return <LoadingSpinner size="small" className="ml-auto shrink-0" />;
}
