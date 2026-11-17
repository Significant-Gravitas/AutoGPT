"use client";

import { BubbleChatIcon } from "@hugeicons/core-free-icons";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/atoms/Button/Button";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { expertKickoffHref } from "../../expertKickoff";

interface Props {
  expertId: string;
  name: string;
}

/** Opens a new expert's first chat, where they introduce themselves; once
 *  that chat exists, the same link opens it again. The chat page only starts
 *  an expert it finds in the roster, so the click refreshes the roster on the
 *  way there. */
export function ExpertChatLink({ expertId, name }: Props) {
  const queryClient = useQueryClient();

  function handleClick() {
    void invalidateExpertRosterQueries(queryClient);
  }

  return (
    <Button
      as="NextLink"
      href={expertKickoffHref(expertId)}
      variant="primary"
      size="row"
      leadingIcon={BubbleChatIcon}
      onClick={handleClick}
      className="min-w-0"
    >
      <span className="truncate">Chat with {name}</span>
    </Button>
  );
}
