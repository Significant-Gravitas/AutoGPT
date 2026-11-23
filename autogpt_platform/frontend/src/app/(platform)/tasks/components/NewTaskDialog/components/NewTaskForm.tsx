"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskDraft } from "../draft";
import { useNewTaskDialog } from "../useNewTaskDialog";
import { DialogActions, OPTIONS_ID } from "./DialogActions";
import { Refusal, TaskOptions } from "./TaskOptions/TaskOptions";

const TITLE_HINT_ID = "new-task-title-hint";

interface Props {
  experts: Expert[];
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
  initial?: TaskDraft;
}

/** The dialog's form. It mounts each time the dialog opens, so it starts
 *  from the kept draft (or the task it copies). */
export function NewTaskForm({
  experts,
  onOpenChange,
  onCreated,
  initial,
}: Props) {
  const form = useNewTaskDialog({ experts, onOpenChange, onCreated, initial });
  const refusal = form.options.refusal;

  return (
    <form
      onSubmit={form.submit}
      className="flex flex-col gap-2 rounded-card border border-border-base bg-surface p-3 shadow-sm transition-colors focus-within:border-border-strong"
    >
      <textarea
        aria-label="What needs doing?"
        aria-describedby={form.isLongTitle ? TITLE_HINT_ID : undefined}
        placeholder="What needs doing?"
        value={form.text}
        onChange={(e) => form.setText(e.target.value)}
        onKeyDown={form.onKeyDown}
        rows={2}
        autoFocus
        dir="auto"
        className="[field-sizing:content] max-h-60 min-h-12 w-full resize-none bg-transparent px-1 text-body text-foreground-ink placeholder:text-foreground-faint focus:outline-none"
      />
      {form.isLongTitle ? (
        <Text as="p" id={TITLE_HINT_ID} variant="small" tone="faint">
          The first part becomes the title; the rest is saved as the
          description.
        </Text>
      ) : null}
      {form.options.isOpen ? (
        <div id={OPTIONS_ID}>
          <TaskOptions state={form.options} />
        </div>
      ) : null}
      {refusal?.field === "form" ? <Refusal text={refusal.text} /> : null}
      <DialogActions form={form} />
    </form>
  );
}
