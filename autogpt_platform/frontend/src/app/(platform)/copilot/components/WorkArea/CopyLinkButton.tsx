"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { Link01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { taskLinkURL } from "@/app/(platform)/tasks/taskLink";
import { useFlash } from "./useFlash";

interface Props {
  task: Task;
}

/** Copy the link that opens this task on Home, beside its ⋯ menu: the
 *  same link the card's menu copies. */
export function CopyLinkButton({ task }: Props) {
  const { flashed, flash } = useFlash();
  const isCopied = flashed === "link";

  async function copy() {
    await navigator.clipboard.writeText(taskLinkURL(task.id));
    flash("link");
  }

  return (
    <Button
      variant="ghost-muted"
      size="icon-xs"
      aria-label={isCopied ? "Link copied" : "Copy link"}
      leadingIcon={isCopied ? Tick02Icon : Link01Icon}
      onClick={() => void copy()}
      className="ml-auto shrink-0"
    />
  );
}
