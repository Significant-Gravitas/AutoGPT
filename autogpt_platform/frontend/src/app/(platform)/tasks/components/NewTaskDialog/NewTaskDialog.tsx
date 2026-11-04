import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ArrowUp02Icon } from "@hugeicons/core-free-icons";
import type { ExpertOption } from "../../data/types";
import { AssigneeChip } from "./components/AssigneeChip/AssigneeChip";
import { useNewTaskDialog } from "./useNewTaskDialog";

const HINT_ID = "new-task-hint";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  experts: ExpertOption[];
  onCreated?: (id: string) => void;
}

export function NewTaskDialog({
  isOpen,
  onOpenChange,
  experts,
  onCreated,
}: Props) {
  const form = useNewTaskDialog({ experts, onOpenChange, onCreated });

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
          <div className="flex items-center justify-between gap-2">
            <AssigneeChip
              value={form.assignee}
              choices={form.choices}
              onChange={form.pickAssignee}
            />
            <span className="flex items-center gap-2">
              {form.isEmpty ? (
                <Text
                  as="span"
                  id={HINT_ID}
                  variant="small"
                  tone="faint"
                  aria-live="polite"
                >
                  Describe the task
                </Text>
              ) : null}
              <Button
                type="submit"
                variant="primary"
                size="icon-sm"
                shape="pill"
                aria-label="Create task"
                leadingIcon={ArrowUp02Icon}
                disabled={!form.canSubmit}
                loading={form.isSubmitting}
                aria-describedby={form.isEmpty ? HINT_ID : undefined}
              />
            </span>
          </div>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
