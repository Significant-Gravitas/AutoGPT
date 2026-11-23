"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import { FOCUS } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { useEditableText } from "./useEditableText";

interface Props {
  icon: IconSvgElement;
  label: string;
  value: string;
  /** Said in place of the words while there are none. */
  placeholder: string;
  /** The API's limit on these words. */
  maxLength: number;
  onSave: (text: string) => void;
}

/** One of the task's own words, as a row of its About block: click to edit
 *  it in place. */
export function EditableField({
  icon,
  label,
  value,
  placeholder,
  maxLength,
  onSave,
}: Props) {
  const field = useEditableText({ value, onSave });

  return (
    <li className="relative flex gap-2.5 px-2 py-1.5 not-first:before:absolute not-first:before:-top-px not-first:before:right-2 not-first:before:left-12 not-first:before:h-px not-first:before:bg-border-hairline">
      <span className="mt-0.5 grid size-[26px] shrink-0 place-items-center rounded-md bg-surface-tray text-foreground-text">
        <Icon icon={icon} size={14} tone="inherit" />
      </span>
      {field.isEditing ? (
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <Text variant="small" as="span" tone="quiet">
            {label}
          </Text>
          <textarea
            aria-label={label}
            value={field.text}
            onChange={(e) => field.setText(e.target.value)}
            onKeyDown={field.onKeyDown}
            onBlur={field.save}
            autoFocus
            rows={3}
            maxLength={maxLength}
            className="w-full min-w-0 resize-none rounded-control bg-surface px-2 py-1 text-body-sm text-foreground-ink shadow-pill focus:ring-2 focus:ring-focus-inbox focus:outline-none"
          />
        </span>
      ) : (
        <button
          type="button"
          onClick={field.start}
          className={cn(
            "-my-1 flex min-w-0 flex-1 flex-col rounded-lg px-1.5 py-1 text-left transition-colors duration-100 ease-out hover:bg-surface-tray",
            FOCUS,
          )}
        >
          <Text variant="small" as="span" tone="quiet">
            {label}
          </Text>
          <Text
            variant="inbox"
            as="span"
            tone={value ? "ink" : "faint"}
            className="text-pretty break-words whitespace-pre-wrap"
          >
            {value || placeholder}
          </Text>
        </button>
      )}
    </li>
  );
}
