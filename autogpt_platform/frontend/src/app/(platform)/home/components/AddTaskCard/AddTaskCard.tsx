"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { AssigneePicker } from "../AssigneePicker/AssigneePicker";
import { useAddTaskCard } from "./useAddTaskCard";

interface Props {
  columnKey: string;
  autoExpand?: boolean;
}

export function AddTaskCard({ columnKey, autoExpand = false }: Props) {
  const {
    isEditing,
    title,
    expertId,
    isPickerOpen,
    isExpertsEnabled,
    isPending,
    inputRef,
    startEditing,
    setIsPickerOpen,
    handleChange,
    handleKeyDown,
    handleSelectExpert,
    handlePickerCloseAutoFocus,
  } = useAddTaskCard({ columnKey, autoExpand });

  if (!isEditing) {
    return (
      <Button
        variant="ghost"
        size="small"
        leftIcon={<Icon icon={PlusSignIcon} size={16} />}
        onClick={startEditing}
        className="w-full justify-start rounded-lg border-transparent px-2 font-normal text-zinc-500 hover:border-transparent hover:bg-zinc-200/60 hover:text-zinc-800"
      >
        Add a task…
      </Button>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-zinc-200 bg-white p-2 shadow-sm">
      <Input
        ref={inputRef}
        id={`new-task-${columnKey}`}
        label="New task"
        hideLabel
        size="small"
        autoFocus
        autoComplete="off"
        placeholder={autoExpand ? "What do you want to get done?" : "New task"}
        value={title}
        readOnly={isPending}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        wrapperClassName="mb-0"
        className="border-transparent px-1 focus:border-transparent focus:ring-0"
      />
      {isExpertsEnabled && (
        <div className="flex items-center justify-between gap-2">
          <AssigneePicker
            expertId={expertId}
            onSelect={handleSelectExpert}
            open={isPickerOpen}
            onOpenChange={setIsPickerOpen}
            onCloseAutoFocus={handlePickerCloseAutoFocus}
          />
          {autoExpand && (
            <Text variant="small" tone="muted">
              Type @ to pick who works on it.
            </Text>
          )}
        </div>
      )}
    </div>
  );
}
