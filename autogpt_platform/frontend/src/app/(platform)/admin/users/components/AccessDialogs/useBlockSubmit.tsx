"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryState } from "nuqs";
import { useState } from "react";
import { BLOCKS_TAB, getErrorMessage, TAB_QUERY_PARAM } from "../../helpers";
import { type CreateBlockArgs, useAccessBlocks } from "../../useAccessBlocks";

interface SubmitArgs extends CreateBlockArgs {
  failureTitle: string;
}

// A 409 (someone blocked it first) is shown in the dialog, other failures as
// a toast; both leave the dialog open with nothing cleared.
export function useBlockSubmit() {
  const { toast } = useToast();
  const { createBlock, isCreating } = useAccessBlocks();
  const [, setTab] = useQueryState(TAB_QUERY_PARAM);
  const [conflict, setConflict] = useState<string | null>(null);

  async function submitBlock({ failureTitle, ...args }: SubmitArgs) {
    try {
      await createBlock(args);
      return true;
    } catch (error) {
      if (getStatus(error) === 409) {
        setConflict(getErrorMessage(error));
      } else {
        toast({
          title: failureTitle,
          description: getErrorMessage(error),
          variant: "destructive",
        });
      }
      return false;
    }
  }

  const viewBlocksAction = (
    <Button
      variant="secondary"
      size="small"
      onClick={() => void setTab(BLOCKS_TAB)}
    >
      View blocks
    </Button>
  );

  return {
    submitBlock,
    isCreating,
    conflict,
    clearConflict: () => setConflict(null),
    viewBlocksAction,
  };
}

function getStatus(error: unknown) {
  return typeof error === "object" && error !== null && "status" in error
    ? (error as { status: unknown }).status
    : undefined;
}
