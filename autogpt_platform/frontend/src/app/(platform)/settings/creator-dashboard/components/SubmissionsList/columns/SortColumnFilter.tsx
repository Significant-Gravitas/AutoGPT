"use client";

import { ArrowDownIcon, ArrowUpIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import { type SortDir, type SortKey } from "../../../helpers";
import { ColumnFilter } from "../../ColumnFilter/ColumnFilter";

interface Props {
  sortKey: SortKey;
  activeKey: SortKey | null;
  activeDir: SortDir;
  onChange: (key: SortKey | null, dir: SortDir) => void;
  ascLabel: string;
  descLabel: string;
}

export function SortColumnFilter({
  sortKey,
  activeKey,
  activeDir,
  onChange,
  ascLabel,
  descLabel,
}: Props) {
  const isActive = activeKey === sortKey;

  function pick(dir: SortDir) {
    onChange(sortKey, dir);
  }

  function clear() {
    onChange(null, activeDir);
  }

  return (
    <ColumnFilter active={isActive} label="Sort" align="start">
      <div className="flex flex-col gap-2">
        <Text variant="small-medium" as="span" className="px-2 text-textBlack">
          Sort
        </Text>
        <SortOption
          icon={<ArrowDownIcon size={14} weight="bold" />}
          label={descLabel}
          active={isActive && activeDir === "desc"}
          onClick={() => pick("desc")}
        />
        <SortOption
          icon={<ArrowUpIcon size={14} weight="bold" />}
          label={ascLabel}
          active={isActive && activeDir === "asc"}
          onClick={() => pick("asc")}
        />
        {isActive ? (
          <Button
            variant="secondary"
            size="small"
            onClick={clear}
            className="self-end"
          >
            Clear
          </Button>
        ) : null}
      </div>
    </ColumnFilter>
  );
}

function SortOption({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
        "ease-[cubic-bezier(0.16,1,0.3,1)] transition-[background-color,color,transform] duration-150",
        "active:scale-[0.98] motion-reduce:transition-none motion-reduce:active:scale-100",
        active
          ? "bg-violet-50 text-violet-700"
          : "text-zinc-700 hover:bg-zinc-100",
      )}
    >
      <span
        className={cn(
          "flex h-5 w-5 items-center justify-center rounded text-zinc-500",
          active && "text-violet-700",
        )}
      >
        {icon}
      </span>
      {label}
    </button>
  );
}
