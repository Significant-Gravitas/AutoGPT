"use client";

import {
  getGetExpertQueryKey,
  useRemoveExpertWorkflow,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import {
  Delete02Icon,
  Folder01Icon,
  MoreHorizontalIcon,
  Settings01Icon,
} from "@hugeicons/core-free-icons";
import { useQueryClient } from "@tanstack/react-query";
import NextLink from "next/link";
import { useState } from "react";
import { workflowNeedsSetup } from "../../helpers";

interface Props {
  workflow: ExpertWorkflowRef;
  expertId: string;
  name: string;
  triggerClassName?: string;
}

export function ExpertWorkflowCardMenu({
  workflow,
  expertId,
  name,
  triggerClassName,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isRemoveOpen, setIsRemoveOpen] = useState(false);
  const { mutateAsync: removeWorkflow, isPending: isRemoving } =
    useRemoveExpertWorkflow();
  const libraryHref = workflow.library_agent_id
    ? `/library/agents/${workflow.library_agent_id}`
    : null;

  async function handleRemove() {
    try {
      await removeWorkflow({ expertId, workflowId: workflow.id });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expertId),
        }),
        invalidateExpertRosterQueries(queryClient),
      ]);
      toast({ title: "Workflow removed", variant: "success" });
      setIsRemoveOpen(false);
    } catch {
      toast({ title: "Couldn't remove workflow", variant: "destructive" });
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="icon"
            size="icon"
            aria-label="More actions"
            className={triggerClassName}
          >
            <Icon icon={MoreHorizontalIcon} size={16} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-[11rem]">
          {libraryHref ? (
            <DropdownMenuItem asChild>
              <NextLink href={libraryHref} className="flex items-center gap-2">
                <Icon
                  icon={
                    workflowNeedsSetup(workflow) ? Settings01Icon : Folder01Icon
                  }
                  size={16}
                />
                {workflowNeedsSetup(workflow) ? "Set up" : "Open in library"}
              </NextLink>
            </DropdownMenuItem>
          ) : null}
          {libraryHref ? <DropdownMenuSeparator /> : null}
          <DropdownMenuItem
            className="flex items-center gap-2 text-red-600 focus:text-red-600"
            onSelect={() => setIsRemoveOpen(true)}
          >
            <Icon icon={Delete02Icon} size={16} />
            Remove from expert
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        controlled={{
          isOpen: isRemoveOpen,
          set: (open) => {
            if (!open && !isRemoving) setIsRemoveOpen(false);
          },
        }}
        className="max-h-[60vh] max-w-md"
        title={`Remove ${name}?`}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="body" className="text-zinc-600">
              This expert will stop running it and its schedule will be removed.
              The workflow stays in your library.
            </Text>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="ghost"
                size="small"
                disabled={isRemoving}
                onClick={() => setIsRemoveOpen(false)}
              >
                Keep
              </Button>
              <Button
                type="button"
                variant="destructive"
                size="small"
                loading={isRemoving}
                onClick={handleRemove}
              >
                Remove
              </Button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
