import { coverClassFor } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Activity01Icon, Alert01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import Image from "next/image";
import type { ExpertRosterStatus } from "../../../helpers";

type CoverStatus = ExpertRosterStatus | "built-in";

/** Only the states that ask something of the user get a badge; "idle" and
 *  "built-in" are the resting state and say nothing the card does not. */
const STATUS_STYLES: Partial<
  Record<
    CoverStatus,
    { label: string; className: string; icon: IconSvgElement }
  >
> = {
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
};

interface Props {
  className?: string;
  color: string | undefined;
  status?: CoverStatus;
  /** Cover picture washed over the colour, so the pastel shows through. */
  art?: string | null;
}

export function ExpertCover({ className, color, status, art }: Props) {
  const statusStyle = status ? (STATUS_STYLES[status] ?? null) : null;

  return (
    <div
      className={cn(
        "relative h-28 w-full overflow-hidden rounded-lg bg-zinc-100",
        coverClassFor(color ?? null),
        className,
      )}
    >
      {art ? (
        <Image
          src={art}
          alt=""
          width={2172}
          height={724}
          sizes="(min-width: 1024px) 33vw, (min-width: 768px) 50vw, 100vw"
          className="absolute inset-x-0 top-1/2 h-auto w-full -translate-y-1/2 opacity-30"
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
