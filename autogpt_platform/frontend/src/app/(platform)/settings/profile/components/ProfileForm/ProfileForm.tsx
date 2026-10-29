"use client";

import { motion, useReducedMotion } from "framer-motion";
import { type ReactNode, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import {
  MAX_BIO_LENGTH,
  type ProfileFormState,
  validateForm,
} from "../../helpers";
import {
  EyeClosedIcon,
  EyeIcon,
  LeftToRightListBulletIcon,
  Link01Icon,
  TextBoldIcon,
  TextItalicIcon,
  TextStrikethroughIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  formState: ProfileFormState;
  errors: ReturnType<typeof validateForm>["errors"];
  onChange: <K extends keyof ProfileFormState>(
    key: K,
    value: ProfileFormState[K],
  ) => void;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

interface MarkdownAction {
  label: string;
  icon: ReactNode;
  before: string;
  after: string;
  placeholder: string;
  block?: boolean;
}

const ACTIONS: ReadonlyArray<MarkdownAction> = [
  {
    label: "Bold",
    icon: <Icon icon={TextBoldIcon} size={16} />,
    before: "**",
    after: "**",
    placeholder: "bold text",
  },
  {
    label: "Italic",
    icon: <Icon icon={TextItalicIcon} size={16} />,
    before: "*",
    after: "*",
    placeholder: "italic text",
  },
  {
    label: "Strikethrough",
    icon: <Icon icon={TextStrikethroughIcon} size={16} />,
    before: "~~",
    after: "~~",
    placeholder: "strikethrough",
  },
  {
    label: "Link",
    icon: <Icon icon={Link01Icon} size={16} />,
    before: "[",
    after: "](https://)",
    placeholder: "link text",
  },
  {
    label: "Bulleted list",
    icon: <Icon icon={LeftToRightListBulletIcon} size={16} />,
    before: "- ",
    after: "",
    placeholder: "list item",
    block: true,
  },
];

export function ProfileForm({ formState, errors, onChange }: Props) {
  const reduceMotion = useReducedMotion();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isPreview, setIsPreview] = useState(false);
  const remaining = MAX_BIO_LENGTH - formState.description.length;
  const counterColor =
    remaining < 0
      ? "text-error"
      : remaining < 30
        ? // TODO(tokens): no exact role (text-amber-600 #D97706)
          "text-amber-600"
        : "text-foreground-subtle";

  function applyAction(action: MarkdownAction) {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const value = formState.description;
    const start = textarea.selectionStart ?? 0;
    const end = textarea.selectionEnd ?? 0;
    const scrollTop = textarea.scrollTop;
    const hasSelection = start !== end;
    const selected = hasSelection ? value.substring(start, end) : "";

    let newValue: string;
    let cursorStart: number;
    let cursorEnd: number;

    if (action.block) {
      const lineStart = value.lastIndexOf("\n", start - 1) + 1;
      const alreadyPrefixed = value
        .substring(lineStart)
        .startsWith(action.before);
      if (alreadyPrefixed) {
        newValue = value;
        cursorStart = start;
        cursorEnd = end;
      } else {
        newValue =
          value.substring(0, lineStart) +
          action.before +
          value.substring(lineStart);
        cursorStart = start + action.before.length;
        cursorEnd = end + action.before.length;
      }
    } else {
      const insertText = selected || action.placeholder;
      newValue =
        value.substring(0, start) +
        action.before +
        insertText +
        action.after +
        value.substring(end);
      if (hasSelection) {
        cursorStart = start + action.before.length;
        cursorEnd = end + action.before.length;
      } else {
        cursorStart = start + action.before.length;
        cursorEnd = cursorStart + insertText.length;
      }
    }

    onChange("description", newValue);
    requestAnimationFrame(() => {
      const node = textareaRef.current;
      if (!node) return;
      node.focus({ preventScroll: true });
      node.setSelectionRange(cursorStart, cursorEnd);
      node.scrollTop = scrollTop;
    });
  }

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.06 }}
      className="flex w-full flex-col gap-2"
    >
      <div className="flex items-center justify-between px-4">
        <Text variant="body-medium" as="span" className="text-foreground-base">
          Bio
        </Text>
        <Text
          variant="small"
          as="span"
          className={`tabular-nums transition-colors duration-150 ${counterColor}`}
        >
          {Math.max(remaining, 0)} left
        </Text>
      </div>
      <div className="flex items-center gap-1 px-2">
        <div
          className={cn(
            "flex items-center gap-1 transition-opacity",
            isPreview && "pointer-events-none opacity-40",
          )}
          aria-hidden={isPreview}
        >
          {ACTIONS.map((action) => (
            <button
              key={action.label}
              type="button"
              aria-label={action.label}
              title={action.label}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => applyAction(action)}
              disabled={isPreview}
              className="inline-flex h-8 w-8 items-center justify-center rounded-full text-foreground-secondary transition-colors hover:bg-surface-hover hover:text-foreground-base"
            >
              {action.icon}
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-pressed={isPreview}
          onClick={() => setIsPreview((v) => !v)}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-foreground-body transition-colors hover:bg-surface-hover hover:text-foreground-base"
        >
          {isPreview ? (
            <Icon icon={EyeClosedIcon} size={14} />
          ) : (
            <Icon icon={EyeIcon} size={14} />
          )}
          {isPreview ? "Edit" : "Preview"}
        </button>
      </div>
      {isPreview ? (
        <div
          className={cn(
            "min-h-[8.75rem] w-full rounded-3xl border border-border-base bg-surface px-4 py-2.5",
            "text-sm leading-5.5 text-foreground-base",
          )}
        >
          {formState.description.trim() ? (
            <div
              className={cn(
                "max-w-none break-words text-sm leading-5.5 text-foreground-base",
                "[&_p]:my-2 [&_p]:first:mt-0 [&_p]:last:mb-0",
                "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5",
                "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5",
                "[&_li]:my-1 [&_li]:pl-1",
                "[&_li>p]:my-0",
                // TODO(tokens): no fitting role for [&_a]:text-purple-600 #6C2EDF; #6C2EDF is brand-hover, a hover fill; no exact role (hover:[&_a]:text-purple-700 #5424AE)
                "[&_a]:text-purple-600 [&_a]:underline [&_a]:hover:text-purple-700",
                "[&_strong]:font-semibold",
                "[&_em]:italic",
                "[&_del]:text-foreground-tertiary [&_del]:line-through",
                // TODO(tokens): no exact text token ([&_code]:text-[0.85em])
                "[&_code]:rounded [&_code]:bg-surface-hover [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em]",
                "[&_blockquote]:border-l-2 [&_blockquote]:border-border-strong [&_blockquote]:pl-3 [&_blockquote]:text-foreground-secondary",
                "[&_h1]:my-2 [&_h1]:text-base [&_h1]:font-semibold",
                "[&_h2]:my-2 [&_h2]:text-base [&_h2]:font-semibold",
                "[&_h3]:my-2 [&_h3]:text-sm [&_h3]:font-semibold",
              )}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {formState.description}
              </ReactMarkdown>
            </div>
          ) : (
            <Text variant="body" as="span" className="text-foreground-subtle">
              Nothing to preview yet.
            </Text>
          )}
        </div>
      ) : (
        <Input
          ref={textareaRef}
          id="profile-bio"
          label="Bio"
          hideLabel
          type="textarea"
          rows={5}
          placeholder="Tell people what you build, the workflows you ship, and what you care about."
          value={formState.description}
          error={errors.description}
          onChange={(e) => onChange("description", e.target.value)}
          // TODO(tokens): no fitting role for hover:scrollbar-thumb-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border) and foreground-disabled (a text colour)
          className="!rounded-3xl !rounded-tr-md scrollbar-thin scrollbar-track-transparent scrollbar-thumb-surface-active hover:scrollbar-thumb-zinc-300"
        />
      )}
    </motion.div>
  );
}
