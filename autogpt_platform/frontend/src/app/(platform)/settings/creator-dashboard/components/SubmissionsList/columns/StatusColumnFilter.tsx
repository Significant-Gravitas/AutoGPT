"use client";

import { CheckIcon } from "@phosphor-icons/react";

import type { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import { STATUS_OPTIONS, STATUS_VISUAL } from "../../../helpers";
import { ColumnFilter } from "../../ColumnFilter/ColumnFilter";

interface Props {
  value: SubmissionStatus[];
  onChange: (next: SubmissionStatus[]) => void;
}

export function StatusColumnFilter({ value, onChange }: Props) {
  function toggle(status: SubmissionStatus) {
    if (value.includes(status)) {
      onChange(value.filter((s) => s !== status));
    } else {
      onChange([...value, status]);
    }
  }

  return (
    <ColumnFilter active={value.length > 0} label="Status" align="start">
      <div className="flex flex-col gap-2">
        <Text variant="small-medium" as="span" className="px-2 text-textBlack">
          Filter by status
        </Text>
        <ul className="flex flex-col gap-1">
          {STATUS_OPTIONS.map((option) => {
            const checked = value.includes(option.value);
            const visual = STATUS_VISUAL[option.value];
            return (
              <li key={option.value}>
                <button
                  type="button"
                  onClick={() => toggle(option.value)}
                  aria-pressed={checked}
                  className={cn(
                    "flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left",
                    "ease-[cubic-bezier(0.16,1,0.3,1)] transition-[background-color,transform] duration-150",
                    "active:scale-[0.98] motion-reduce:transition-none motion-reduce:active:scale-100",
                    checked ? "bg-violet-50" : "hover:bg-zinc-100",
                  )}
                >
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${visual.pillClass}`}
                  >
                    {option.label}
                  </span>
                  <span
                    className={cn(
                      "flex h-4 w-4 items-center justify-center rounded-[4px] border",
                      "ease-[cubic-bezier(0.16,1,0.3,1)] transition-[background-color,border-color,transform] duration-200",
                      checked
                        ? "scale-100 border-violet-600 bg-violet-600 text-white"
                        : "scale-95 border-zinc-300 bg-white text-transparent",
                    )}
                  >
                    <CheckIcon size={10} weight="bold" />
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
        {value.length > 0 ? (
          <Button
            variant="secondary"
            size="small"
            onClick={() => onChange([])}
            className="self-end"
          >
            Clear
          </Button>
        ) : null}
      </div>
    </ColumnFilter>
  );
}
