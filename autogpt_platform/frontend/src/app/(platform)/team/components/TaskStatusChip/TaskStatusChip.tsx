import { DelegatedTaskStatus } from "@/app/api/__generated__/models/delegatedTaskStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Orb } from "@/components/atoms/Orb/Orb";
import type { OrbVariant } from "@/components/atoms/Orb/helpers";
import { cn } from "@/lib/utils";
import {
  getStatusIcon,
  getStatusIconClass,
  getStatusLabel,
} from "../../task-helpers";

interface Props {
  status: DelegatedTaskStatus;
  /** Which orb sweep the working state runs. */
  orbVariant?: OrbVariant;
  /** Tailwind text-color class for the working orb — an expert's accent, so
   *  a busy row reads as *that* expert working. Neutral when omitted. */
  accentClassName?: string;
}

export function TaskStatusChip({
  status,
  orbVariant,
  accentClassName,
}: Props) {
  const icon = getStatusIcon(status);

  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-zinc-700">
      {icon ? (
        <Icon icon={icon} size={16} className={getStatusIconClass(status)} />
      ) : (
        <Orb
          variant={orbVariant}
          size={20}
          label={getStatusLabel(status)}
          className={cn("text-zinc-800", accentClassName)}
        />
      )}
      {getStatusLabel(status)}
    </span>
  );
}
