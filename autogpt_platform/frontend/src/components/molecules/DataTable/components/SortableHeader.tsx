import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowUp01Icon,
  UnfoldMoreIcon,
} from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import type { SortDirection } from "../helpers";

interface Props {
  children: ReactNode;
  direction: SortDirection | null;
  align: "left" | "center" | "right";
  onSort: () => void;
}

export function SortableHeader({ children, direction, align, onSort }: Props) {
  const icon =
    direction === "asc"
      ? ArrowUp01Icon
      : direction === "desc"
        ? ArrowDown01Icon
        : UnfoldMoreIcon;

  return (
    <button
      type="button"
      onClick={onSort}
      className={cn(
        "-mx-1.5 inline-flex items-center gap-1 rounded-md px-1.5 py-1 font-sans text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2",
        direction && "text-zinc-900",
        align === "right" && "flex-row-reverse",
      )}
    >
      {children}
      <Icon
        icon={icon}
        size={14}
        aria-hidden
        className={cn(!direction && "text-zinc-400")}
      />
    </button>
  );
}
