"use client";

import { Page } from "@/app/(platform)/pages/components/Page/Page";
import { useListPages } from "@/app/api/__generated__/endpoints/pages/pages";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import type { ToolUIPart } from "ai";
import NextLink from "next/link";
import { useState } from "react";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { GenericTool } from "../GenericTool/GenericTool";
import {
  APPLY_MESSAGE,
  DISCARD_MESSAGE,
  isOutdated,
  parsePageOutput,
} from "./helpers";

interface Props {
  part: ToolUIPart;
  canAct: boolean;
  isCurrentlyStreaming?: boolean;
}

export function DesignPage({ part, canAct, isCurrentlyStreaming }: Props) {
  const { onSend } = useCopilotChatActions();
  const [isSent, setIsSent] = useState(false);
  const output = parsePageOutput(part);
  const stored = useListPages({
    query: { enabled: Boolean(output?.page.id), select: okData },
  });

  if (!output) {
    const isPending =
      part.state === "input-streaming" || part.state === "input-available";
    if (isPending && isCurrentlyStreaming) {
      return <Skeleton className="my-2 h-48 w-full rounded-2xl" />;
    }
    return <GenericTool part={part} />;
  }

  const outdated = isOutdated(output.page, stored.data?.pages);

  function send(message: string) {
    if (!canAct || isSent || outdated) return;
    setIsSent(true);
    void onSend(message);
  }

  return (
    <div
      className="my-3 flex w-full min-w-0 flex-col gap-3 rounded-2xl bg-surface-muted p-4"
      data-testid="page-card"
    >
      <Text
        variant="small-medium"
        tone="secondary"
        transform="uppercase"
        tracking="caps"
      >
        {output.saved ? "Page saved" : "Page preview"}
      </Text>
      <Page page={output.page} />
      {outdated ? (
        <Text variant="body" tone="secondary" role="status">
          A newer version of this page exists, so this preview can&apos;t be
          applied.
        </Text>
      ) : null}
      {output.saved && output.page.id ? (
        <NextLink href={`/pages/${output.page.id}`} className="w-fit underline">
          <Text as="span" variant="body-medium">
            Open it in Pages
          </Text>
        </NextLink>
      ) : !output.saved && !outdated ? (
        <div className="flex gap-2">
          <Button
            size="small"
            disabled={!canAct || isSent}
            onClick={() => send(APPLY_MESSAGE)}
          >
            Apply
          </Button>
          <Button
            size="small"
            variant="outline"
            disabled={!canAct || isSent}
            onClick={() => send(DISCARD_MESSAGE)}
          >
            Discard
          </Button>
        </div>
      ) : null}
    </div>
  );
}
