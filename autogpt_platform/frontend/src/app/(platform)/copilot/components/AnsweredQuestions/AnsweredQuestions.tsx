import { Text } from "@/components/atoms/Text/Text";

export interface AnsweredQuestion {
  key: string;
  question: string;
  answer?: string;
}

interface Props {
  items: AnsweredQuestion[];
}

/** The questions the user has moved past, one quiet line each where they
 *  were asked. The answer itself is the user's message below. */
export function AnsweredQuestions({ items }: Props) {
  return (
    <ul className="flex flex-col gap-0.5">
      {items.map((item) => (
        <li key={item.key}>
          <Text variant="inbox" tone="quiet" className="text-pretty">
            Asked: {item.question}
            <Text as="span" variant="inbox" tone="faint">
              {item.answer ? " · answered below" : " · put off"}
            </Text>
          </Text>
        </li>
      ))}
    </ul>
  );
}

interface AnswerProps {
  answers: { question: string; answer: string }[];
}

/** The user's answers as they read in the thread: what was asked in a small
 *  line, then the answer in the user's own bubble. */
export function AnswerBubbles({ answers }: AnswerProps) {
  return (
    <div className="ml-auto flex max-w-[80%] flex-col items-end gap-2">
      {answers.map((line) => (
        <div
          key={line.question}
          className="flex max-w-full flex-col items-end gap-1"
        >
          <Text variant="inbox" tone="faint" className="text-right text-pretty">
            Answered: {line.question}
          </Text>
          <div className="max-w-full rounded-2xl bg-foreground-black/5 px-3.5 py-2 text-body leading-relaxed [overflow-wrap:anywhere] whitespace-pre-wrap text-foreground-ink">
            {line.answer}
          </div>
        </div>
      ))}
    </div>
  );
}
