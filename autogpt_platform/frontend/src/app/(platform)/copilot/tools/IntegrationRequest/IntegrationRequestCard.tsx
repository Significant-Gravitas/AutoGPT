"use client";

import type { ToolUIPart } from "ai";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Unlink04Icon } from "@hugeicons/core-free-icons";
import { GenericTool } from "../GenericTool/GenericTool";
import { parseIntegrationRequest } from "./helpers";

interface Props {
  part: ToolUIPart;
}

/** Otto's answer when a service can't be connected yet: said once, plainly,
 *  where the user asked, so the reply below can go straight to what works. */
export function IntegrationRequestCard({ part }: Props) {
  const request = parseIntegrationRequest(part.output);
  if (!request) {
    // Still running: the card arrives with the answer. Settled on anything
    // else (an error, or a service that turned out to be supported): show it
    // the way any other tool result is shown rather than drop it.
    const settled =
      part.state === "output-available" || part.state === "output-error";
    return settled ? <GenericTool part={part} /> : null;
  }
  const { service, recorded } = request;
  return (
    <div className="my-2 flex items-start gap-3 rounded-3xl border border-separator bg-surface px-4 py-3.5">
      <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-surface-hover">
        <Icon
          icon={Unlink04Icon}
          size={16}
          className="text-foreground-secondary"
        />
      </span>
      <div className="flex min-w-0 flex-col gap-1 [overflow-wrap:anywhere]">
        <Text variant="body-medium" className="text-foreground-strong">
          {service} isn&apos;t supported yet
        </Text>
        <Text variant="small" className="text-foreground-secondary">
          Otto can&apos;t connect to {service}, so it can&apos;t do this step
          for you.{" "}
          {recorded
            ? "We've added your request to the list of connections people want next."
            : "Otto can still help with the rest of the task."}
        </Text>
      </div>
    </div>
  );
}
