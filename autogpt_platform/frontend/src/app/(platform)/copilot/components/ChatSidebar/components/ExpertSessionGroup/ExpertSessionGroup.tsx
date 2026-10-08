import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState, type ReactNode } from "react";
import { EXPERT_CHAT_PAGE_SIZE } from "@/services/experts/expert-chat-pagination";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";

interface Props {
  groupKey: string;
  label: string;
  role?: string | null;
  sessions: SessionSummaryResponse[];
  renderRow: (
    session: SessionSummaryResponse,
    index: number,
    list: SessionSummaryResponse[],
  ) => ReactNode;
}

export function ExpertSessionGroup({
  groupKey,
  label,
  role,
  sessions,
  renderRow,
}: Props) {
  const [visibleCount, setVisibleCount] = useState(EXPERT_CHAT_PAGE_SIZE);
  const visible = sessions.slice(0, visibleCount);
  const hiddenCount = sessions.length - visible.length;
  const headerId = `session-group-${groupKey}`;

  return (
    <Collapsible
      defaultOpen
      role="group"
      aria-labelledby={headerId}
      className="group/collapsible flex flex-col gap-1"
    >
      <CollapsibleTrigger
        id={headerId}
        data-testid={`expert-group-header-${groupKey}`}
        className="flex items-center justify-between gap-2 px-3 pt-2 pb-1 text-muted-foreground hover:text-zinc-700"
      >
        {groupKey === "pinned" ? (
          <Text as="span" variant="body-medium">
            {label}
          </Text>
        ) : (
          <ExpertIdentityDetails
            isOtto={groupKey === "autopilot"}
            name={label}
            role={role}
            size="compact"
          />
        )}
        <Icon
          icon={ArrowDown01Icon}
          className="size-4 transition-transform duration-200 group-data-open/collapsible:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-1">
        {visible.map((session, index) => renderRow(session, index, visible))}
        {hiddenCount > 0 && (
          <Button
            variant="ghost"
            size="md"
            data-testid={`expert-group-load-more-${groupKey}`}
            onClick={() =>
              setVisibleCount((count) => count + EXPERT_CHAT_PAGE_SIZE)
            }
          >
            Load more
          </Button>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
