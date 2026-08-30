import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import { useAnswerTask } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useToast } from "@/components/molecules/Toast/use-toast";

interface Props {
  item: HomeAttentionItem;
}

export function EscalationAnswer({ item }: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [text, setText] = useState("");

  const { mutate: answerTask, isPending } = useAnswerTask({
    mutation: {
      onSuccess: async () => {
        toast({ title: "Answer sent — the task is resuming" });
        await queryClient.invalidateQueries({
          queryKey: getGetHomeDashboardQueryKey(),
        });
      },
      onError: () => {
        toast({
          title: "Could not send your answer",
          description: "Try again.",
          variant: "destructive",
        });
      },
    },
  });

  function submit(answer: string) {
    if (!item.task_id || !answer.trim() || isPending) return;
    answerTask({ taskId: item.task_id, data: { answer: answer.trim() } });
  }

  const options = item.options ?? [];

  return (
    <div className="mt-2 flex flex-col gap-2">
      {options.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {options.map((option) => (
            <Button
              key={option}
              variant="secondary"
              size="small"
              disabled={isPending}
              onClick={() => submit(option)}
            >
              {option}
            </Button>
          ))}
        </div>
      ) : null}
      <form
        className="flex items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit(text);
        }}
      >
        <Input
          id={`escalation-answer-${item.id}`}
          label="Your answer"
          hideLabel
          size="small"
          placeholder="Type an answer…"
          value={text}
          onChange={(event) => setText(event.target.value)}
          disabled={isPending}
          wrapperClassName="mb-0 flex-1"
        />
        <Button
          type="submit"
          variant="primary"
          size="small"
          loading={isPending}
          disabled={!text.trim()}
        >
          Answer
        </Button>
      </form>
    </div>
  );
}
