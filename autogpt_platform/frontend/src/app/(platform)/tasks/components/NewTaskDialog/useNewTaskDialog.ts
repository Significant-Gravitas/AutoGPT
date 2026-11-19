import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import {
  buildCreateRequest,
  OTTO_VALUE,
  ownerChoices,
  resolveChoice,
  savedForLaterTitle,
} from "./helpers";
import { rhythmOf } from "./repeat";
import { isTimed, optionsRequest } from "./taskOptions";
import { useCreateAndStart } from "./useCreateAndStart";
import { useTaskOptions } from "./useTaskOptions";

interface Args {
  experts: Expert[];
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
}

export function useNewTaskDialog({ experts, onOpenChange, onCreated }: Args) {
  const [text, setText] = useState("");
  const [picked, setPicked] = useState(OTTO_VALUE);
  const options = useTaskOptions();
  const creation = useCreateAndStart();
  const choices = ownerChoices(experts);
  const owner = resolveChoice(choices, picked);
  const isEmpty = text.trim().length === 0;
  const canSubmit = !isEmpty && !creation.isCreating;
  const timed = isTimed(options.options);

  /** Closing forgets what was typed, who was picked and the options. */
  function close() {
    onOpenChange(false);
    setText("");
    setPicked(OTTO_VALUE);
    options.reset();
  }

  /** Create the task and, unless it waits for a time or is saved for
   *  later, start it. The dialog closes as soon as the task exists. */
  async function submit(e?: FormEvent, start = true) {
    e?.preventDefault();
    if (!canSubmit) return;
    const label = owner.label;
    const { repeat } = options.options;
    const created = await creation.create({
      ...buildCreateRequest(text, owner),
      ...optionsRequest(options.options),
    });
    if (created.refusal) return options.refuse(created.refusal);
    if (!created.task) return;
    close();
    onCreated?.(created.task.id);
    if (start && !timed) return void creation.start(created.task.id, label);
    toast({
      title: savedForLaterTitle(
        label,
        created.task,
        repeat ? rhythmOf(repeat) : null,
        Date.now(),
      ),
    });
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (!isKey(e, "Enter") || e.shiftKey) return;
    e.preventDefault();
    void submit();
  }

  return {
    text,
    setText,
    choices,
    owner,
    pickOwner: setPicked,
    options,
    isTimed: timed,
    canSubmit,
    isEmpty,
    isSubmitting: creation.isCreating,
    submit,
    saveForLater: () => void submit(undefined, false),
    onKeyDown,
    close,
  };
}
