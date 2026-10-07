"use client";

import { motion, useReducedMotion } from "framer-motion";
import { type ReactNode, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/atoms/Button/Button";
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
      ? "text-red-500"
      : remaining < 30
        ? "text-yellow-600"
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
            <Button
              key={action.label}
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={action.label}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => applyAction(action)}
              disabled={isPreview}
              className="rounded-full text-zinc-600 hover:bg-zinc-100 hover:text-black"
            >
              {action.icon}
            </Button>
          ))}
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-pressed={isPreview}
          onClick={() => setIsPreview((v) => !v)}
          leadingIcon={isPreview ? EyeClosedIcon : EyeIcon}
          className="ml-auto h-8 rounded-full px-3 text-zinc-700 hover:bg-zinc-100 hover:text-black"
        >
          {isPreview ? "Edit" : "Preview"}
        </Button>
      </div>
      {isPreview ? (
        <div
          className={cn(
            "min-h-35 w-full rounded-3xl border border-zinc-200 bg-white px-4 py-2.5",
            "text-sm leading-[22px] text-black",
          )}
        >
          {formState.description.trim() ? (
            <div
              className={cn(
                "max-w-none text-sm leading-[22px] wrap-break-word text-black",
                "[&_p]:my-2 [&_p]:first:mt-0 [&_p]:last:mb-0",
                "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5",
                "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5",
                "[&_li]:my-1 [&_li]:pl-1",
                "[&_li>p]:my-0",
                "[&_a]:text-purple-600 [&_a]:underline [&_a]:hover:text-purple-700",
                "[&_strong]:font-semibold",
                "[&_em]:italic",
                "[&_del]:text-muted-foreground [&_del]:line-through",
                "[&_code]:rounded-sm [&_code]:bg-zinc-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em]",
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
          className="scrollbar-thin scrollbar-thumb-zinc-200 scrollbar-track-transparent rounded-3xl rounded-tr-md hover:scrollbar-thumb-zinc-300"
        />
      )}
    </motion.div>
  );
}
