import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { isKey } from "@/lib/keyboard";
import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { readDraft, writeDraft, type TaskDraft } from "./draft";
import {
  buildCreateRequest,
  isLongFirstLine,
  OTTO_VALUE,
  ownerChoices,
  resolveChoice,
  savedForLaterTitle,
} from "./helpers";
import { rhythmOf } from "./repeat";
import { isTimed, optionsRequest } from "./taskOptions";
import { useCreateAndStart } from "./useCreateAndStart";
import { useTaskOptions } from "./useTaskOptions";

const PICK_A_TIME = "Pick a start time, or choose Now.";

interface Args {
  experts: Expert[];
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
  /** A task to start from (Duplicate, a suggestion). Not kept as a draft. */
  initial?: TaskDraft;
}

/** The create form. An unsent draft (text, owner, options) is kept for the
 *  browser session, so leaving the page and coming back finds it again;
 *  it goes once the task is created or the draft is discarded. One submit
 *  at a time: a double click or a second Enter sends one task. */
export function useNewTaskDialog({
  experts,
  onOpenChange,
  onCreated,
  initial,
}: Args) {
  const userID = useAuthStore((s) => s.user?.id);
  const keepsDraft = !initial;
  const [seed] = useState(() => initial ?? readDraft(userID));
  const [text, setText] = useState(seed?.text ?? "");
  const [picked, setPicked] = useState(seed?.owner ?? OTTO_VALUE);
  const options = useTaskOptions(seed?.options);
  const creation = useCreateAndStart();
  const submitting = useRef(false);
  const choices = ownerChoices(experts);
  const owner = resolveChoice(choices, picked);
  const isEmpty = text.trim().length === 0;
  const canSubmit = !isEmpty && !creation.isCreating;
  const timed = isTimed(options.options);

  useEffect(() => {
    if (!keepsDraft) return;
    writeDraft(
      userID,
      text.trim() ? { text, owner: picked, options: options.options } : null,
    );
  }, [keepsDraft, userID, text, picked, options.options]);

  function close() {
    onOpenChange(false);
  }

  /** Throw the draft away and close. */
  function discard() {
    writeDraft(userID, null);
    onOpenChange(false);
  }

  async function submit(e?: FormEvent, start = true) {
    e?.preventDefault();
    if (!canSubmit || submitting.current) return;
    if (options.options.start?.kind === "picking") {
      return options.refuse(PICK_A_TIME);
    }
    submitting.current = true;
    try {
      await send(start);
    } finally {
      submitting.current = false;
    }
  }

  /** Create the task and, unless it waits for a time or is saved for
   *  later, start it. The dialog closes as soon as the task exists. */
  async function send(start: boolean) {
    const label = owner.label;
    const { repeat } = options.options;
    const created = await creation.create(
      {
        ...buildCreateRequest(text, owner),
        ...optionsRequest(options.options, Date.now()),
      },
      () => void submit(undefined, start),
    );
    if (created.refusal) return options.refuse(created.refusal);
    if (!created.task) return;
    if (keepsDraft) writeDraft(userID, null);
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
    isLongTitle: isLongFirstLine(text),
    canSubmit,
    canDiscard: keepsDraft && !isEmpty,
    isEmpty,
    isSubmitting: creation.isCreating,
    submit,
    saveForLater: () => void submit(undefined, false),
    onKeyDown,
    close,
    discard,
  };
}

export type NewTaskForm = ReturnType<typeof useNewTaskDialog>;
