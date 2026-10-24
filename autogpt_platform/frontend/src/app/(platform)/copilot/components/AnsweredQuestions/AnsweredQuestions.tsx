import { MessageQuestionIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export interface AnsweredQuestion {
  key: string;
  question: string;
  answer?: string;
}

interface Props {
  items: AnsweredQuestion[];
}

/** A question card the user has moved past, kept in the thread so they can
 *  reread what they were asked and what they answered. */
export function AnsweredQuestions({ items }: Props) {
  return (
    <div className="flex w-full max-w-xl flex-col gap-2.5 rounded-xl bg-surface p-3 ring-1 ring-border-base/70">
      {items.map((item) => (
        <div key={item.key} className="flex items-start gap-2">
          <Icon
            icon={MessageQuestionIcon}
            size={14}
            className="mt-0.5 shrink-0 text-foreground-subtle"
          />
          <div className="min-w-0">
            <p className="text-body-sm leading-inherit text-foreground-tertiary">
              {item.question}
            </p>
            {item.answer && (
              <p className="whitespace-pre-wrap text-body-sm leading-inherit text-foreground-emphasis">
                {item.answer}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
