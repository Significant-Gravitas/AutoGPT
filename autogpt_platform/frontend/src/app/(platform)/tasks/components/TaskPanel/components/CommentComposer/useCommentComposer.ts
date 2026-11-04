import { type ChangeEvent, type KeyboardEvent, useRef, useState } from "react";
import { isKey } from "@/lib/keyboard";
import type { ExpertOption } from "../../../../data/types";
import { useCommentOnTask } from "../../../../data/useTasksData";
import {
  type MentionQuery,
  filterExperts,
  findMentionQuery,
  insertMention,
} from "./helpers";

export function useCommentComposer(taskId: string, experts: ExpertOption[]) {
  const [value, setValue] = useState("");
  const [mention, setMention] = useState<MentionQuery | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const comment = useCommentOnTask();

  const suggestions = mention ? filterExperts(experts, mention.query) : [];
  const isMenuOpen = suggestions.length > 0;

  function updateMention(text: string, caret: number) {
    const next = findMentionQuery(text, caret);
    setMention(next);
    setActiveIndex(0);
  }

  function handleChange(event: ChangeEvent<HTMLTextAreaElement>) {
    const text = event.target.value;
    setValue(text);
    updateMention(text, event.target.selectionStart ?? text.length);
  }

  function selectExpert(expert: ExpertOption) {
    if (!mention) return;
    const next = insertMention(value, mention, expert.name);
    setValue(next.value);
    setMention(null);
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(next.caret, next.caret);
    });
  }

  function submit() {
    const body = value.trim();
    if (!body || comment.isPending) return;
    comment.mutate(
      { id: taskId, body },
      {
        onSuccess: () => {
          setValue("");
          setMention(null);
        },
      },
    );
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (isMenuOpen) {
      if (isKey(event, "ArrowDown")) {
        event.preventDefault();
        setActiveIndex((i) => (i + 1) % suggestions.length);
        return;
      }
      if (isKey(event, "ArrowUp")) {
        event.preventDefault();
        setActiveIndex(
          (i) => (i - 1 + suggestions.length) % suggestions.length,
        );
        return;
      }
      if (
        (isKey(event, "Enter") || isKey(event, "Tab")) &&
        !event.metaKey &&
        !event.ctrlKey
      ) {
        event.preventDefault();
        selectExpert(
          suggestions[Math.min(activeIndex, suggestions.length - 1)],
        );
        return;
      }
      if (isKey(event, "Escape")) {
        event.preventDefault();
        event.stopPropagation();
        setMention(null);
        return;
      }
    }
    if (isKey(event, "Enter") && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submit();
    }
  }

  return {
    value,
    textareaRef,
    handleChange,
    handleKeyDown,
    handleBlur: () => setMention(null),
    suggestions,
    isMenuOpen,
    activeIndex,
    selectExpert,
    submit,
    isSending: comment.isPending,
    isError: comment.isError,
  };
}
