import { coverClassFor } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  Activity01Icon,
  Alert01Icon,
  Clock01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import Image from "next/image";
import type { ExpertRosterStatus } from "../../../helpers";

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
  status?: CoverStatus;
  /** Show Autopilot's cover art. Implied by the "built-in" status, so a
   *  caller that wants the art without the badge sets this instead. */
  builtIn?: boolean;
}

export function ExpertCover({ className, color, status, builtIn }: Props) {
  const statusStyle = status ? STATUS_STYLES[status] : null;
  const showArt = builtIn || status === "built-in";

  return (
    <div
      className={cn(
        "relative h-28 w-full overflow-hidden rounded-lg bg-zinc-100",
        coverClassFor(color ?? null),
        className,
      )}
    >
      {showArt ? (
        <Image
          src="/experts/covers/autopilot.jpg"
          alt=""
          fill
          sizes="(min-width: 1024px) 33vw, (min-width: 768px) 50vw, 100vw"
          className="object-cover"
        />
      ) : null}
      {statusStyle ? (
        <Text
          variant="small-medium"
          as="span"
          className={cn(
            "absolute bottom-3 right-3 flex items-center gap-1 rounded-md px-2 py-0.5",
            statusStyle.className,
          )}
        >
          <Icon icon={statusStyle.icon} size={13} />
          {statusStyle.label}
        </Text>
      ) : null}
    </div>
  );
}
