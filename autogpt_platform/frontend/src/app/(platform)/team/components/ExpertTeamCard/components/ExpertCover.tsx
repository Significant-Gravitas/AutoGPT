import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Activity01Icon,
  Alert01Icon,
  Clock01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import Image from "next/image";
import { type ExpertRosterStatus, getExpertCoverSrc } from "../../../helpers";

type CoverStatus = ExpertRosterStatus | "built-in";

const STATUS_STYLES: Record<
  CoverStatus,
  { label: string; className: string; icon: IconSvgElement }
> = {
  idle: {
    label: "Idle",
    className: "bg-white text-zinc-700",
    icon: Clock01Icon,
  },
  working: {
    label: "Working",
    className: "bg-emerald-50 text-emerald-700",
    icon: Activity01Icon,
  },
  "needs-you": {
    label: "Needs you",
    className: "bg-amber-50 text-amber-700",
    icon: Alert01Icon,
  },
  "built-in": {
    label: "Built in",
    className: "bg-white text-zinc-700",
    icon: SparklesIcon,
  },
};

interface Props {
  className?: string;
  color: string | undefined;
  src?: string;
  status?: CoverStatus;
}

export function ExpertCover({ className, color, src, status }: Props) {
  const statusStyle = status ? STATUS_STYLES[status] : null;

  return (
    <div
      className={cn(
        "relative h-32 w-full overflow-hidden rounded-[1.75rem] bg-zinc-100",
        className,
      )}
    >
      <Image
        src={src ?? getExpertCoverSrc(color)}
        alt=""
        fill
        sizes="(min-width: 1024px) 33vw, (min-width: 768px) 50vw, 100vw"
        className="object-cover"
      />
      {statusStyle ? (
        <span
          className={`absolute bottom-3 right-3 flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium ${statusStyle.className}`}
        >
          <Icon icon={statusStyle.icon} size={13} />
          {statusStyle.label}
        </span>
      ) : null}
    </div>
  );
}
