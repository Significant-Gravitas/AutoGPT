"use client";

import { ArrowRight01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { useDelegationQuestionBox } from "./useDelegationQuestionBox";

interface Props {
  question: string;
  options: string[];
  expertName: string;
  /** The chat card sends with a short "Send"; the panel names who hears it. */
  variant: "chain" | "panel";
  isSending: boolean;
  onSend: (answer: string) => void;
}

/** A teammate's question with its option chips and a free-text answer. The
 *  same box answers from the chat's wire and from the Work panel. */
export function DelegationQuestionBox({
  question,
  options,
  expertName,
  variant,
  isSending,
  onSend,
}: Props) {
  const { picked, pick, text, setText, answer, send, handleKeyDown } =
    useDelegationQuestionBox(onSend);
  const inputId = `delegation-answer-${variant}`;

  return (
    <div className="flex flex-col gap-2.5" data-testid="delegation-question">
      <p className="whitespace-pre-wrap text-sm leading-[22px] text-zinc-900">
        {question}
      </p>
      {options.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          {options.map((option) => (
            <Button
              key={option}
              size="xs"
              variant={picked === option ? "primary" : "secondary"}
              aria-pressed={picked === option}
              className="rounded-full"
              leadingIcon={picked === option ? Tick02Icon : undefined}
              onClick={() => pick(option)}
            >
              {option}
            </Button>
          ))}
        </div>
      )}
      <div
        className={cn(
          "flex gap-2",
          variant === "panel" ? "flex-col" : "flex-col sm:flex-row",
        )}
      >
        <Input
          id={inputId}
          label={`Answer ${expertName}`}
          hideLabel
          size="small"
          placeholder={
            options.length > 0 ? "Or answer in your own words…" : "Answer…"
          }
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          wrapperClassName="mb-0 min-w-0 flex-1"
        />
        <Button
          size="small"
          variant="primary"
          loading={isSending}
          disabled={!answer || isSending}
          onClick={send}
          className="shrink-0 self-start"
          rightIcon={
            variant === "panel" ? (
              <Icon icon={ArrowRight01Icon} size={14} />
            ) : undefined
          }
        >
          {variant === "panel" ? `Send to ${expertName}` : "Send"}
        </Button>
      </div>
    </div>
  );
}
