"use client";

import { SentIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ExpertOption } from "../../../../data/types";
import { AssigneeAvatar } from "../AssigneeAvatar/AssigneeAvatar";
import { useCommentComposer } from "./useCommentComposer";

interface Props {
  taskId: string;
  experts: ExpertOption[];
}

export function CommentComposer({ taskId, experts }: Props) {
  const {
    value,
    textareaRef,
    handleChange,
    handleKeyDown,
    handleBlur,
    suggestions,
    isMenuOpen,
    activeIndex,
    selectExpert,
    submit,
    isSending,
    isError,
  } = useCommentComposer(taskId, experts);
  const listboxId = `task-mentions-${taskId}`;

  return (
    <div className="flex flex-col gap-2">
      <div className="relative">
        {isMenuOpen ? (
          <ul
            id={listboxId}
            role="listbox"
            aria-label="Mention an expert"
            className="absolute bottom-full left-0 z-10 mb-1.5 flex w-64 animate-inbox-enter flex-col gap-0.5 rounded-card bg-surface p-1 shadow-card"
          >
            {suggestions.map((expert, index) => (
              <li
                key={expert.id}
                id={`${listboxId}-${expert.id}`}
                role="option"
                aria-selected={index === activeIndex}
                onMouseDown={(event) => {
                  event.preventDefault();
                  selectExpert(expert);
                }}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-control px-2 py-1.5 transition-[background-color] duration-150 ease-out-expo",
                  index === activeIndex
                    ? "bg-surface-wash"
                    : "hover:bg-surface-wash",
                )}
              >
                <AssigneeAvatar
                  assignee={{
                    type: "expert",
                    id: expert.id,
                    name: expert.name,
                    avatar_url: expert.avatar_url,
                  }}
                  size={18}
                />
                <Text variant="inbox-medium" tone="ink" as="span">
                  {expert.name}
                </Text>
              </li>
            ))}
          </ul>
        ) : null}
        <textarea
          ref={textareaRef}
          aria-label="Comment"
          role="combobox"
          aria-expanded={isMenuOpen}
          aria-controls={isMenuOpen ? listboxId : undefined}
          aria-autocomplete="list"
          aria-activedescendant={
            isMenuOpen && suggestions[activeIndex]
              ? `${listboxId}-${suggestions[activeIndex].id}`
              : undefined
          }
          placeholder="Leave a comment, @ to mention an expert"
          rows={2}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          readOnly={isSending}
          className="block min-h-16 w-full resize-none rounded-card bg-surface px-3 py-2.5 text-body-sm text-foreground-ink shadow-card transition-shadow duration-150 ease-out-expo placeholder:text-foreground-faint focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <Text variant="small" tone={isError ? "error" : "faint"}>
          {isError
            ? "Your comment could not be posted. Try again."
            : "⌘/Ctrl + Enter to send"}
        </Text>
        <Button
          variant="primary"
          size="row"
          leadingIcon={SentIcon}
          loading={isSending}
          disabled={!value.trim()}
          onClick={submit}
        >
          Comment
        </Button>
      </div>
    </div>
  );
}
