"use client";

import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useTaskOutcomeReview } from "./useTaskOutcomeReview";

interface Props {
  task: DelegatedTask;
}

export function TaskOutcomeReview({ task }: Props) {
  const {
    isRequestingChanges,
    note,
    setNote,
    handleAccept,
    revealNote,
    cancelNote,
    submitChanges,
    isAccepting,
    isRejecting,
  } = useTaskOutcomeReview(task.id);

  if (task.status !== "DONE") return null;

  if (task.acceptance === "ACCEPTED") {
    return (
      <div className="flex items-center gap-1.5 text-emerald-600">
        <Icon icon={CheckmarkCircle02Icon} size={16} />
        <Text variant="small" className="text-emerald-700">
          Outcome accepted
        </Text>
      </div>
    );
  }

  // A rejection under the cap reopens the task (status leaves DONE), so a
  // REJECTED verdict only survives here once the revision cap is hit.
  if (task.acceptance === "REJECTED") {
    return (
      <Text variant="small" className="text-zinc-500">
        You have asked for changes twice — Autopilot escalated this to a chat so
        you can clarify what you need.
      </Text>
    );
  }

  if (isRequestingChanges) {
    return (
      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submitChanges();
        }}
      >
        <Input
          id={`task-outcome-note-${task.id}`}
          label="What should change?"
          hideLabel
          size="small"
          placeholder="What should change?"
          value={note}
          onChange={(event) => setNote(event.target.value)}
          disabled={isRejecting}
          wrapperClassName="mb-0"
        />
        <div className="flex items-center gap-2">
          <Button
            type="submit"
            variant="primary"
            size="small"
            loading={isRejecting}
            disabled={!note.trim()}
          >
            Send changes
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="small"
            disabled={isRejecting}
            onClick={cancelNote}
          >
            Cancel
          </Button>
        </div>
      </form>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <Button
        variant="primary"
        size="small"
        loading={isAccepting}
        onClick={handleAccept}
      >
        Accept
      </Button>
      <Button
        variant="secondary"
        size="small"
        disabled={isAccepting}
        onClick={revealNote}
      >
        Request changes
      </Button>
    </div>
  );
}
