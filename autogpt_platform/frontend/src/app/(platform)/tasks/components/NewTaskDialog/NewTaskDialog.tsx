import type { Expert } from "@/app/api/__generated__/models/expert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { DialogActions, OPTIONS_ID } from "./components/DialogActions";
import { Refusal, TaskOptions } from "./components/TaskOptions/TaskOptions";
import { useNewTaskDialog } from "./useNewTaskDialog";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  experts: Expert[];
  onCreated?: (id: string) => void;
}

/** Write a task, pick who takes it, and, under Options, when it starts or
 *  how often it runs and what done looks like. */
export function NewTaskDialog({
  isOpen,
  onOpenChange,
  experts,
  onCreated,
}: Props) {
  const form = useNewTaskDialog({ experts, onOpenChange, onCreated });
  const refusal = form.options.refusal;

  return (
    <Dialog
      title="New task"
      className="max-w-xl max-lg:inset-x-0 max-lg:mx-auto"
      controlled={{
        isOpen,
        set: (open) => (open ? onOpenChange(true) : form.close()),
      }}
    >
      <Dialog.Content>
        <form
          onSubmit={form.submit}
          className="flex flex-col gap-2 rounded-card border border-border-base bg-surface p-3 shadow-sm transition-colors focus-within:border-border-strong"
        >
          <textarea
            aria-label="What needs doing?"
            placeholder="What needs doing?"
            value={form.text}
            onChange={(e) => form.setText(e.target.value)}
            onKeyDown={form.onKeyDown}
            rows={2}
            autoFocus
            className="[field-sizing:content] max-h-60 min-h-12 w-full resize-none bg-transparent px-1 text-body text-foreground-ink placeholder:text-foreground-faint focus:outline-none"
          />
          {form.options.isOpen ? (
            <div id={OPTIONS_ID}>
              <TaskOptions state={form.options} />
            </div>
          ) : null}
          {refusal?.field === "form" ? <Refusal text={refusal.text} /> : null}
          <DialogActions form={form} />
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
