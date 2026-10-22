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
    <div className="flex w-full max-w-xl flex-col gap-2.5 rounded-xl bg-white p-3 ring-1 ring-zinc-200/70">
      {items.map((item) => (
        <div key={item.key} className="flex items-start gap-2">
          <Icon
            icon={MessageQuestionIcon}
            size={14}
            className="mt-0.5 shrink-0 text-zinc-400"
          />
          <div className="min-w-0">
            <p className="text-[13px] text-zinc-500">{item.question}</p>
            {item.answer && (
              <p className="whitespace-pre-wrap text-[13px] text-zinc-800">
                {item.answer}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
