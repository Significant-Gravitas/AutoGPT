"use client";

import {
  getGetTaskQueryKey,
  useAnswerTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Props {
  taskId: string;
  options: string[];
}

/** Answer an escalation right where it's read — one-click options when the
 *  expert offered them, free text otherwise. Mirrors the Home screen's
 *  EscalationAnswer so the two surfaces behave identically. */
export function ActivityAnswer({ taskId, options }: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [text, setText] = useState("");

  const { mutate: answerTask, isPending } = useAnswerTask({
    mutation: {
      onSuccess: async () => {
        toast({ title: "Answer sent — the task is resuming" });
        await queryClient.invalidateQueries({
          queryKey: getGetTaskQueryKey(taskId),
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
    if (!answer.trim() || isPending) return;
    answerTask({ taskId, data: { answer: answer.trim() } });
  }

  return (
    <div className="mt-2.5 flex max-w-2xl flex-col gap-2">
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
          id={`task-activity-answer-${taskId}`}
          label="Your answer"
          hideLabel
          size="small"
          placeholder="Reply to this question…"
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
