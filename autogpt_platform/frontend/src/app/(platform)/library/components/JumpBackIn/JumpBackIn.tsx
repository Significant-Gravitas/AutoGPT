"use client";

import { ArrowRight, Lightning } from "@phosphor-icons/react";
import NextLink from "next/link";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useJumpBackIn } from "./useJumpBackIn";

export function JumpBackIn() {
  const { agent, isLoading } = useJumpBackIn();

  if (isLoading || !agent) {
    return null;
  }

  return (
    <div className="rounded-large bg-gradient-to-r from-zinc-200 via-zinc-100 to-zinc-100/40 p-[1px]">
      <div className="flex items-center justify-between rounded-large bg-[#F6F7F8] px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-zinc-900">
            <Lightning size={18} weight="fill" className="text-white" />
          </div>
          <div className="flex flex-col">
            <Text variant="small" className="text-zinc-500">
              Continue where you left off
            </Text>
            <Text variant="body-medium" className="text-zinc-900">
              {agent.name}
            </Text>
          </div>
        </div>
        <NextLink href={`/library/agents/${agent.id}`}>
          <Button variant="primary" size="small" className="gap-1.5">
            Jump Back In
            <ArrowRight size={16} />
          </Button>
        </NextLink>
      </div>
    </div>
  );
}
