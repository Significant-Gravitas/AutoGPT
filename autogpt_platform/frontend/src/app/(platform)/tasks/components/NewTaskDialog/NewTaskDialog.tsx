import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { ExpertOption } from "../../data/types";
import { assigneeOptions, isPriority, PRIORITY_OPTIONS } from "./helpers";
import { useNewTaskDialog } from "./useNewTaskDialog";

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
  const form = useNewTaskDialog({ onOpenChange, onCreated });

  return (
    <Dialog
      title="New task"
      className="max-w-lg"
      controlled={{
        isOpen,
        set: (open) => (open ? onOpenChange(true) : form.close()),
      }}
    >
      <Dialog.Content>
        <form className="flex flex-col gap-1" onSubmit={form.submit}>
          <Input
            id="new-task-title"
            label="Title"
            placeholder="What needs doing?"
            value={form.title}
            onChange={(e) => form.setTitle(e.target.value)}
            autoFocus
          />
          <Input
            id="new-task-description"
            label="Description"
            type="textarea"
            rows={3}
            placeholder="Add details, links or the result you expect"
            value={form.description}
            onChange={(e) => form.setDescription(e.target.value)}
          />
          <Select
            id="new-task-assignee"
            label="Assignee"
            value={form.assignee}
            onValueChange={form.setAssignee}
            options={assigneeOptions(experts)}
          />
          <div className="grid grid-cols-1 gap-x-3 sm:grid-cols-2">
            <Select
              id="new-task-priority"
              label="Priority"
              value={form.priority}
              onValueChange={(value) => {
                if (isPriority(value)) form.setPriority(value);
              }}
              options={PRIORITY_OPTIONS}
            />
            <Input
              id="new-task-due"
              label="Due date"
              type="date"
              value={form.dueDate}
              onChange={(e) => form.setDueDate(e.target.value)}
            />
          </div>
          <Dialog.Footer>
            <Button
              type="button"
              variant="secondary"
              size="small"
              onClick={form.close}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              size="small"
              disabled={!form.canSubmit}
              loading={form.isSubmitting}
            >
              Create task
            </Button>
          </Dialog.Footer>
        </form>
      </Dialog.Content>
    </Dialog>
  );
}
