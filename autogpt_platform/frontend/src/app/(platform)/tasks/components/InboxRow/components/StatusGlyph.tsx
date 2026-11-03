import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Alert02Icon,
  Loading03Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { InboxState } from "../../../data/types";

interface Props {
  state: InboxState;
}

export function StatusGlyph({ state }: Props) {
  return (
    <span
      aria-hidden
      className="inline-flex size-4 shrink-0 items-center justify-center"
    >
      {state === "running" ? (
        <Icon icon={Loading03Icon} size={14} tone="info" spin />
      ) : null}
      {state === "waiting" ? (
        <span className="size-2 rounded-full bg-status-waiting" />
      ) : null}
      {state === "done" ? <Icon icon={Tick02Icon} size={14} tone="ok" /> : null}
      {state === "failed" ? (
        <Icon icon={Alert02Icon} size={14} tone="error" />
      ) : null}
    </span>
  );
}
