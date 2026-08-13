import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  HelpCircleIcon,
  Loading03Icon,
  PauseIcon,
  Settings01Icon,
} from "@hugeicons/core-free-icons";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

const STATUS_CONFIG = {
  working: {
    label: "Working",
    icon: Loading03Icon,
    className: "bg-primary/10 text-primary",
  },
  ready: {
    label: "Ready",
    icon: CheckmarkCircle02Icon,
    className: "bg-emerald-50 text-emerald-700",
  },
  needs_setup: {
    label: "Setup",
    icon: Settings01Icon,
    className: "bg-amber-50 text-amber-700",
  },
  paused: {
    label: "Paused",
    icon: PauseIcon,
    className: "bg-zinc-100 text-zinc-600",
  },
  failed: {
    label: "Failed",
    icon: AlertCircleIcon,
    className: "bg-rose-50 text-rose-700",
  },
};

const UNKNOWN_STATUS_CONFIG = {
  label: "Unknown",
  icon: HelpCircleIcon,
  className: "bg-zinc-100 text-zinc-600",
};

interface Props {
  status: HomeAgentStatus["status"];
}

export function StatusBadge({ status }: Props) {
  const config = STATUS_CONFIG[status] ?? UNKNOWN_STATUS_CONFIG;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-medium",
        config.className,
      )}
    >
      <Icon
        icon={config.icon}
        size={13}
        className={cn(status === "working" && "animate-spin")}
        aria-hidden="true"
      />
      {config.label}
    </span>
  );
}
