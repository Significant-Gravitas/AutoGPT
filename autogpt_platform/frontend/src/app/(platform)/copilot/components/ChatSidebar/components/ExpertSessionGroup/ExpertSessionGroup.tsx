import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState, type ReactNode } from "react";

export const INITIAL_VISIBLE_SESSIONS = 5;
const LOAD_MORE_INCREMENT = 10;

interface Props {
  groupKey: string;
  label: string;
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
  sessions,
  renderRow,
}: Props) {
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_SESSIONS);
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
        className="flex items-center justify-between px-3 pb-1 pt-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-700"
      >
        {label}
        <Icon
          icon={ArrowDown01Icon}
          className="size-4 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-180 motion-reduce:transition-none"
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-1 overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down motion-reduce:animate-none">
        {visible.map((session, index) => renderRow(session, index, visible))}
        {hiddenCount > 0 && (
          <Button
            variant="ghost"
            size="small"
            data-testid={`expert-group-load-more-${groupKey}`}
            onClick={() =>
              setVisibleCount((count) => count + LOAD_MORE_INCREMENT)
            }
          >
            Load more
          </Button>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
