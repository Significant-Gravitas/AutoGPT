"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import Link from "next/link";
import { useRemoveTriggerAgent } from "../../../../hooks/useRemoveTriggerAgent";
import { MoreVerticalIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  getBuilderHref,
  getLibraryAgentHref,
} from "@/services/org-team/builder";

interface Props {
  parentAgent: LibraryAgent;
  triggerAgent: LibraryAgent;
  onDeleted?: () => void;
}

export function TriggerAgentActionsDropdown({
  parentAgent,
  triggerAgent,
  onDeleted,
}: Props) {
  const { openDialog, dialog } = useRemoveTriggerAgent({
    parentAgent,
    triggerAgent,
    onDeleted,
  });

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="icon"
            size="icon"
            onClick={(e) => e.stopPropagation()}
            aria-label="More actions"
            className="ml-auto min-w-fit shrink-0"
          >
            <Icon icon={MoreVerticalIcon} size={18} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <Link
              href={getLibraryAgentHref(
                triggerAgent.id,
                triggerAgent.organization_id ?? null,
                triggerAgent.team_id ?? null,
              )}
              onClick={(e) => e.stopPropagation()}
            >
              View in library
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link
              href={getBuilderHref({
                graphId: triggerAgent.graph_id,
                graphVersion: triggerAgent.graph_version,
                organizationId: triggerAgent.organization_id ?? null,
                teamId: triggerAgent.team_id ?? null,
              })}
              onClick={(e) => e.stopPropagation()}
            >
              Open in builder
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              openDialog();
            }}
          >
            Remove trigger
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {dialog}
    </>
  );
}
