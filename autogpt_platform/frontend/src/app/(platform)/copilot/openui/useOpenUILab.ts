import { useState, type FormEvent, type KeyboardEvent } from "react";
import { BuiltinActionType, type ActionEvent } from "@openuidev/react-lang";
import {
  getSampleResponse,
  scenarios,
  type Scenario,
} from "@/lib/openui/scenarios";
import { isKey } from "@/lib/keyboard";
import { useWorkspaceGeneration } from "./useWorkspaceGeneration";
import { getActionFields } from "./helpers";

export interface LabMessage {
  role: "user" | "assistant";
  text: string;
}

function initialMessages(scenario: Scenario): LabMessage[] {
  return [
    { role: "user", text: scenario.prompt },
    {
      role: "assistant",
      text: "Here's an interactive workspace to explore. You can work with the results directly, or ask for the next view.",
    },
  ];
}

export function useOpenUILab(liveAvailable: boolean) {
  const [scenario, setScenario] = useState<Scenario>(scenarios[0]);
  const [mode, setMode] = useState<"sample" | "live">("sample");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<LabMessage[]>(
    initialMessages(scenarios[0]),
  );
  const generation = useWorkspaceGeneration(scenarios[0].source);

  async function send(message: string, fields: Record<string, unknown> = {}) {
    if (!message.trim() || generation.isStreaming) return;
    const sample =
      mode === "sample" ? getSampleResponse(message, fields) : null;
    if (mode === "sample" && !sample) {
      generation.setError(
        "Sample mode replays prepared examples. Try a suggested follow-up, or switch to Live AI for your own requests.",
      );
      return;
    }
    if (mode === "live" && !liveAvailable) {
      generation.setError(
        "Live AI is not configured in this environment. Sample mode is ready to explore.",
      );
      return;
    }
    setMessages((current) => [
      ...current.slice(-6),
      { role: "user", text: message },
    ]);
    setPrompt("");
    if (await generation.generate(message, sample, fields)) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text:
            mode === "sample"
              ? "The next sample is ready. Try the controls in your workspace."
              : "Your workspace is ready. You can refine it with another request.",
        },
      ]);
    }
  }

  function selectScenario(next: Scenario) {
    setScenario(next);
    setMode("sample");
    setPrompt("");
    setMessages(initialMessages(next));
    generation.reset(next.source);
    void generation.generate(next.prompt, next.source);
  }

  function changeMode(next: "sample" | "live") {
    generation.stop();
    generation.setError(null);
    setMode(next);
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void send(prompt);
  }
  function handleKeyDown(
    event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(event, "Enter") && !event.shiftKey) {
      event.preventDefault();
      void send(prompt);
    }
  }
  function handleAction(event: ActionEvent) {
    if (event.type === BuiltinActionType.ContinueConversation)
      void send(
        event.humanFriendlyMessage,
        getActionFields(event.formState, event.formName),
      );
  }
  function replay() {
    void generation.generate("Replay sample", generation.source);
  }

  return {
    scenario,
    mode,
    prompt,
    setPrompt,
    messages,
    ...generation,
    send,
    selectScenario,
    changeMode,
    handleSubmit,
    handleKeyDown,
    handleAction,
    replay,
  };
}
