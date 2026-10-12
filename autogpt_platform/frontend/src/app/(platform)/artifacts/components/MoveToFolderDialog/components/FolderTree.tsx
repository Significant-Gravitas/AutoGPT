"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { isKey } from "@/lib/keyboard";
import {
  ArrowRight01Icon,
  Folder01Icon,
  Home01Icon,
} from "@hugeicons/core-free-icons";
import type { KeyboardEvent } from "react";
import { useRef } from "react";
import { FOLDER_STYLE } from "../../WorkspaceFolders/folder-constants";
import { rowKey, type TreeRow } from "../helpers";

interface Props {
  rows: TreeRow[];
  selectedKey: string | null;
  onSelect: (id: string | null) => void;
  onToggleExpanded: (folderId: string) => void;
}

/**
 * WAI-ARIA tree of move destinations. A click selects rather than moves —
 * with a chevron beside every name, a mis-click must not move anything.
 */
export function FolderTree({
  rows,
  selectedKey,
  onSelect,
  onToggleExpanded,
}: Props) {
  const listRef = useRef<HTMLUListElement>(null);
  // One row in the tab order: the selected one, else the first.
  const tabbableIndex = Math.max(
    0,
    rows.findIndex((row) => rowKey(row.id) === selectedKey),
  );

  function focusRow(index: number) {
    const clamped = Math.min(Math.max(index, 0), rows.length - 1);
    const items =
      listRef.current?.querySelectorAll<HTMLLIElement>('[role="treeitem"]');
    items?.[clamped]?.focus();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLLIElement>, index: number) {
    const row = rows[index];
    if (isKey(e, "ArrowDown")) {
      e.preventDefault();
      focusRow(index + 1);
    } else if (isKey(e, "ArrowUp")) {
      e.preventDefault();
      focusRow(index - 1);
    } else if (isKey(e, "Home")) {
      e.preventDefault();
      focusRow(0);
    } else if (isKey(e, "End")) {
      e.preventDefault();
      focusRow(rows.length - 1);
    } else if (isKey(e, "ArrowRight")) {
      if (!row.hasChildren) return;
      e.preventDefault();
      if (row.isExpanded) focusRow(index + 1);
      else if (row.id) onToggleExpanded(row.id);
    } else if (isKey(e, "ArrowLeft")) {
      if (!row.isExpanded) return;
      e.preventDefault();
      if (row.id) onToggleExpanded(row.id);
    } else if (isKey(e, "Enter", " ")) {
      e.preventDefault();
      if (!row.disabledReason) onSelect(row.id);
    }
  }

  return (
    <ul
      ref={listRef}
      role="tree"
      aria-label="Destination folders"
      className="max-h-[20rem] overflow-y-auto py-1"
    >
      {rows.map((row, index) => {
        const key = rowKey(row.id);
        const isSelected = key === selectedKey;
        const isDisabled = row.disabledReason !== null;
        return (
          <li
            key={key}
            role="treeitem"
            aria-level={row.level}
            aria-selected={isSelected}
            aria-disabled={isDisabled || undefined}
            aria-expanded={row.hasChildren ? row.isExpanded : undefined}
            tabIndex={index === tabbableIndex ? 0 : -1}
            onKeyDown={(e) => handleKeyDown(e, index)}
            onClick={() => !isDisabled && onSelect(row.id)}
            style={{ paddingLeft: `${(row.level - 1) * 20 + 8}px` }}
            className={cn(
              "flex items-center gap-2 rounded-lg py-2 pr-2 outline-none",
              isDisabled
                ? "cursor-not-allowed text-zinc-400"
                : "cursor-pointer text-zinc-800 hover:bg-zinc-50",
              isSelected && "bg-zinc-100",
              "focus-visible:ring-2 focus-visible:ring-zinc-300",
            )}
            data-testid="move-to-folder-option"
          >
            {row.hasChildren && row.id ? (
              <button
                type="button"
                tabIndex={-1}
                aria-label={`${row.isExpanded ? "Collapse" : "Expand"} ${row.name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleExpanded(row.id as string);
                }}
                className="flex h-[18px] w-[22px] shrink-0 items-center justify-center rounded text-zinc-400 hover:bg-zinc-200 hover:text-zinc-600"
              >
                <Icon
                  icon={ArrowRight01Icon}
                  size={14}
                  className={cn(
                    "transition-transform",
                    row.isExpanded && "rotate-90",
                  )}
                />
              </button>
            ) : (
              <span className="w-[22px] shrink-0" aria-hidden />
            )}
            <Icon
              icon={row.id === null ? Home01Icon : Folder01Icon}
              size={18}
              className={cn(
                "shrink-0",
                isDisabled ? "text-zinc-300" : FOLDER_STYLE.icon,
              )}
            />
            <Text
              variant="small-medium"
              as="span"
              className="min-w-0 truncate"
              title={row.name}
            >
              {row.name}
            </Text>
            {row.disabledReason ? (
              <Text
                variant="small"
                as="span"
                className="ml-auto shrink-0 pl-2 text-zinc-400"
              >
                {row.disabledReason}
              </Text>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
