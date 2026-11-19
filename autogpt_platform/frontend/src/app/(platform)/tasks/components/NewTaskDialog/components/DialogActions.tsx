import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowDown01Icon, ArrowUp02Icon } from "@hugeicons/core-free-icons";
import type { useNewTaskDialog } from "../useNewTaskDialog";
import { OwnerPicker } from "./OwnerPicker/OwnerPicker";

export const HINT_ID = "new-task-hint";
export const OPTIONS_ID = "new-task-options";

interface Props {
  form: ReturnType<typeof useNewTaskDialog>;
}

/** The dialog's bottom row: who does it, the options toggle, and save or
 *  create. A task that waits for a time is created, not started. */
export function DialogActions({ form }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2">
      <span className="flex items-center gap-1">
        <OwnerPicker
          value={form.owner}
          choices={form.choices}
          onChange={form.pickOwner}
        />
        <Button
          type="button"
          variant={form.isTimed ? "secondary" : "ghost-muted"}
          size="xs"
          shape="pill"
          aria-expanded={form.options.isOpen}
          aria-controls={OPTIONS_ID}
          onClick={form.options.toggle}
          rightIcon={<Icon icon={ArrowDown01Icon} size={12} tone="faint" />}
        >
          Options
        </Button>
      </span>
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
        {form.isTimed ? null : (
          <Button
            type="button"
            variant="ghost-muted"
            size="xs"
            shape="pill"
            disabled={!form.canSubmit}
            onClick={form.saveForLater}
          >
            Save for later
          </Button>
        )}
        <Button
          type="submit"
          variant="primary"
          size="icon-sm"
          shape="pill"
          aria-label={form.isTimed ? "Create task" : "Create and start task"}
          leadingIcon={ArrowUp02Icon}
          disabled={!form.canSubmit}
          loading={form.isSubmitting}
          aria-describedby={form.isEmpty ? HINT_ID : undefined}
        />
      </span>
    </div>
  );
}
