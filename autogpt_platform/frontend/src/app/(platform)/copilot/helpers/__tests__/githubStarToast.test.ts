import { toast } from "@/components/molecules/Toast/use-toast";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildKickoffMessage } from "../../expertKickoff";
import {
  countAnswerTowardStarToast,
  resetStarToastCounts,
} from "../githubStarToast";

vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast: vi.fn() }));

const EXPERT_ID = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b";

function textMessage(
  id: string,
  role: UIMessage["role"],
  text: string,
): UIMessage {
  return { id, role, parts: [{ type: "text", text }] };
}

const kickoff = buildKickoffMessage(EXPERT_ID);
const kickoffPrompt: UIMessage = {
  ...textMessage("kickoff", "user", kickoff.text),
  metadata: kickoff.metadata,
};
const reply = textMessage("reply", "assistant", "Hi, I'm your new analyst.");

let promptNumber = 0;

function newPrompt() {
  promptNumber += 1;
  return textMessage(`prompt-${promptNumber}`, "user", "Plan my week");
}

function answer(messages: UIMessage[], times: number) {
  for (let turn = 0; turn < times; turn++) {
    countAnswerTowardStarToast({ messages });
  }
}

function answerNewPrompts(times: number, before: UIMessage[] = []) {
  for (let turn = 0; turn < times; turn++) {
    countAnswerTowardStarToast({ messages: [...before, newPrompt(), reply] });
  }
}

// Only answers to prompts the user sent count toward the GitHub star toast.
describe("countAnswerTowardStarToast", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    window.localStorage.clear();
    resetStarToastCounts();
    vi.mocked(toast).mockClear();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("doesn't count an expert's kickoff, which the app sends itself", () => {
    answer([kickoffPrompt, reply], 5);
    expect(toast).not.toHaveBeenCalled();

    answerNewPrompts(5, [kickoffPrompt, reply]);
    expect(toast).toHaveBeenCalledOnce();
  });

  it("doesn't count a turn that answers no prompt", () => {
    answer([reply], 5);
    expect(toast).not.toHaveBeenCalled();
  });

  it("counts a prompt once, however many turns answer it", () => {
    answer([newPrompt(), reply], 5);
    expect(toast).not.toHaveBeenCalled();

    answerNewPrompts(4);
    expect(toast).toHaveBeenCalledOnce();
  });
});
