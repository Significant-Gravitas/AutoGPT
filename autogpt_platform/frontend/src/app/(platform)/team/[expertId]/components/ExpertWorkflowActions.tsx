"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Button } from "@/components/atoms/Button/Button";
import { PencilEdit02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { ExpertWorkflowCardMenu } from "./ExpertWorkflowCardMenu";

interface Props {
  workflow: ExpertWorkflowRef;
  expertId?: string;
  name: string;
  builderHref: string | null;
  chatHref: string;
  chatPrompt: string;
  /** Hosts with an inline chat open it with the prompt instead of leaving
   *  for the Copilot page. */
  onAsk?: (prompt: string) => void;
  /** Cards float these over the cover art; the list rows sit on white. */
  variant?: "floating" | "ghost";
  size?: "icon-xs" | "icon-sm";
}

export function ExpertWorkflowActions({
  workflow,
  expertId,
  name,
  builderHref,
  chatHref,
  chatPrompt,
  onAsk,
  variant = "floating",
  size = "icon-xs",
}: Props) {
  return (
    <>
      {builderHref ? (
        <Button
          as="NextLink"
          href={builderHref}
          variant={variant}
          size={size}
          aria-label="Edit workflow"
          leadingIcon={PencilEdit02Icon}
        />
      ) : null}
      {onAsk ? (
        <Button
          type="button"
          variant={variant}
          size={size}
          aria-label="Ask about this workflow"
          leadingIcon={SparklesIcon}
          onClick={() => onAsk(chatPrompt)}
        />
      ) : (
        <Button
          as="NextLink"
          href={chatHref}
          variant={variant}
          size={size}
          aria-label="Ask about this workflow"
          leadingIcon={SparklesIcon}
        />
      )}
      <ExpertWorkflowCardMenu
        workflow={workflow}
        expertId={expertId}
        name={name}
        variant={variant}
        size={size}
      />
    </>
  );
}
