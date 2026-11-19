"use client";

import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowUpRight01Icon, File02Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";

interface Props {
  text: string | null;
  isFailure: boolean;
  /** The document the run wrote, when the task still holds it. */
  document: TaskArtifact | null;
  chatHref: string;
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

const LINK =
  "inline-flex w-fit items-center gap-1 rounded-sm text-body-sm font-medium text-foreground-text underline-offset-2 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-focus-inbox";

/** A run opened in the Runs list: what came out of it, in words, then the
 *  document it wrote and the chat it ran in. */
export function RunResult({
  text,
  isFailure,
  document,
  chatHref,
  onOpenArtifact,
}: Props) {
  return (
    <li className="flex flex-col gap-2 pt-0.5 pr-2 pb-3 pl-[2.875rem]">
      <Text
        variant="inbox"
        tone={isFailure ? "error" : "text"}
        className="text-pretty whitespace-pre-line"
      >
        {text ?? "This run left no result."}
      </Text>
      <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {document ? (
          <button
            type="button"
            className={LINK}
            onClick={() => onOpenArtifact(document)}
          >
            <Icon icon={File02Icon} size={14} tone="inherit" />
            {document.title}
          </button>
        ) : null}
        <NextLink href={chatHref} className={LINK}>
          Open the chat it ran in
          <Icon icon={ArrowUpRight01Icon} size={14} tone="inherit" />
        </NextLink>
      </span>
    </li>
  );
}
