import { useState, type FormEvent, type KeyboardEvent } from "react";
import { BuiltinActionType, type ActionEvent } from "@openuidev/react-lang";
import {
  getSampleResponse,
  scenarios,
  type Scenario,
} from "@/lib/openui/scenarios";
import { isKey } from "@/lib/keyboard";
import { useWorkspaceGeneration } from "./useWorkspaceGeneration";
import { getActionFields } from "@/lib/openui/actions";

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

export function useOpenUILab() {
  const [scenario, setScenario] = useState<Scenario>(scenarios[0]);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<LabMessage[]>(
    initialMessages(scenarios[0]),
  );
  const generation = useWorkspaceGeneration(scenarios[0].source);

  async function send(message: string, fields: Record<string, unknown> = {}) {
    if (!message.trim() || generation.isStreaming) return;
    const sample = getSampleResponse(message, fields);
    if (!sample) {
      generation.setError(
        "This preview replays prepared examples. Try a suggested follow-up, or continue in Copilot for your own requests.",
      );
      return;
    }

    setMessages((current) => [
      ...current.slice(-6),
      { role: "user", text: message },
    ]);
    setPrompt("");
    if (await generation.generate(sample)) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: "The next sample is ready. Try the controls in your workspace.",
        },
      ]);
    }
  }

  function selectScenario(next: Scenario) {
    setScenario(next);
    setPrompt("");
    setMessages(initialMessages(next));
    generation.reset(next.source);
    void generation.generate(next.source);
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
    void generation.generate(generation.source);
  }

  return {
    scenario,
    prompt,
    setPrompt,
    messages,
    ...generation,
    send,
    selectScenario,
    handleSubmit,
    handleKeyDown,
    handleAction,
    replay,
  };
}
