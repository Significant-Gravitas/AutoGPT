import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { CheckIcon, CircleIcon, PauseCircleIcon } from "@phosphor-icons/react";
import type { DisplayStatus } from "../../helpers";

interface Props {
  status: DisplayStatus;
}

export function StatusIcon({ status }: Props) {
  if (status === "completed") {
    return (
      <CheckIcon
        size={14}
        weight="bold"
        className="text-emerald-500"
        aria-label="completed"
      />
    );
  }
  if (status === "in_progress") {
    return (
      <LoadingSpinner
        size="small"
        className="h-3.5 w-3.5 text-purple-500 [animation-duration:0.5s]"
        aria-label="in progress"
      />
    );
  }
  if (status === "stopped") {
    return (
      <PauseCircleIcon
        size={15}
        weight="fill"
        className="text-amber-500"
        aria-label="stopped"
      />
    );
  }
  return (
    <CircleIcon
      size={14}
      weight="regular"
      className="text-zinc-400"
      aria-label="pending"
    />
  );
}
