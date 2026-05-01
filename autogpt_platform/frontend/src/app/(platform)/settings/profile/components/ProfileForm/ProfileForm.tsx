"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  EyeClosedIcon,
  EyeIcon,
  LinkIcon,
  ListBulletsIcon,
  TextBIcon,
  TextItalicIcon,
  TextStrikethroughIcon,
} from "@phosphor-icons/react";
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
    icon: <TextBIcon size={16} weight="bold" />,
    before: "**",
    after: "**",
    placeholder: "bold text",
  },
  {
    label: "Italic",
    icon: <TextItalicIcon size={16} />,
    before: "*",
    after: "*",
    placeholder: "italic text",
  },
  {
    label: "Strikethrough",
    icon: <TextStrikethroughIcon size={16} />,
    before: "~~",
    after: "~~",
    placeholder: "strikethrough",
  },
  {
    label: "Link",
    icon: <LinkIcon size={16} />,
    before: "[",
    after: "](https://)",
    placeholder: "link text",
  },
  {
    label: "Bulleted list",
    icon: <ListBulletsIcon size={16} />,
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
      ? "text-red-500"
      : remaining < 30
        ? "text-amber-600"
        : "text-zinc-400";

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
        <Text variant="body-medium" as="span" className="text-black">
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
              className="inline-flex h-8 w-8 items-center justify-center rounded-full text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-black"
            >
              {action.icon}
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-pressed={isPreview}
          onClick={() => setIsPreview((v) => !v)}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-black"
        >
          {isPreview ? <EyeClosedIcon size={14} /> : <EyeIcon size={14} />}
          {isPreview ? "Edit" : "Preview"}
        </button>
      </div>
      {isPreview ? (
        <div
          className={cn(
            "min-h-[8.75rem] w-full rounded-3xl border border-zinc-200 bg-white px-4 py-2.5",
            "text-sm leading-[22px] text-black",
          )}
        >
          {formState.description.trim() ? (
            <div
              className={cn(
                "max-w-none break-words text-sm leading-[22px] text-black",
                "[&_p]:my-2 first:[&_p]:mt-0 last:[&_p]:mb-0",
                "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5",
                "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5",
                "[&_li]:my-1 [&_li]:pl-1",
                "[&_li>p]:my-0",
                "[&_a]:text-purple-600 [&_a]:underline hover:[&_a]:text-purple-700",
                "[&_strong]:font-semibold",
                "[&_em]:italic",
                "[&_del]:text-zinc-500 [&_del]:line-through",
                "[&_code]:rounded [&_code]:bg-zinc-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em]",
                "[&_blockquote]:border-l-2 [&_blockquote]:border-zinc-300 [&_blockquote]:pl-3 [&_blockquote]:text-zinc-600",
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
            <Text variant="body" as="span" className="text-zinc-400">
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
          placeholder="Tell people what you build, the agents you ship, and what you care about."
          value={formState.description}
          error={errors.description}
          onChange={(e) => onChange("description", e.target.value)}
          className="!rounded-3xl !rounded-tr-md scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200 hover:scrollbar-thumb-zinc-300"
        />
      )}
    </motion.div>
  );
}
