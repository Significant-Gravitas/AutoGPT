"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import Link from "next/link";
import { useRemoveTriggerAgent } from "../../../hooks/useRemoveTriggerAgent";
import {
  Delete02Icon,
  LinkSquare01Icon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  parentAgent: LibraryAgent;
  triggerAgent: LibraryAgent;
  onDeleted?: () => void;
}

export function SelectedTriggerAgentActions({
  parentAgent,
  triggerAgent,
  onDeleted,
}: Props) {
  const { openDialog, isDeleting, dialog } = useRemoveTriggerAgent({
    parentAgent,
    triggerAgent,
    onDeleted,
  });

  return (
    <>
      <div className="my-4 flex flex-col items-center gap-3">
        <Link
          href={`/library/agents/${triggerAgent.id}`}
          aria-label="View in library"
        >
          <Button variant="icon" size="icon" aria-label="View in library">
            <Icon icon={LinkSquare01Icon} size={18} className="text-zinc-700" />
          </Button>
        </Link>
        <Link
          href={`/build?flowID=${triggerAgent.graph_id}`}
          aria-label="Open in builder"
        >
          <Button variant="icon" size="icon" aria-label="Open in builder">
            <Icon icon={PencilIcon} size={18} className="text-zinc-700" />
          </Button>
        </Link>
        <Button
          variant="icon"
          size="icon"
          aria-label="Remove trigger"
          onClick={openDialog}
          disabled={isDeleting}
        >
          {isDeleting ? (
            <LoadingSpinner size="small" />
          ) : (
            <Icon icon={Delete02Icon} size={18} />
          )}
        </Button>
      </div>
      {dialog}
    </>
  );
}
