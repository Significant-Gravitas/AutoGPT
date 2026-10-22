import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Mic01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { joinNames, type IntroAsk } from "../helpers";

interface Props {
  ask: NonNullable<IntroAsk>;
  disabled: boolean;
  onConnect: () => void;
  onDismissConnect: () => void;
  onAnswer: (message: string) => void;
  onStartVoice: () => void;
  onType: () => void;
}

export function AskCard({
  ask,
  disabled,
  onConnect,
  onDismissConnect,
  onAnswer,
  onStartVoice,
  onType,
}: Props) {
  if (ask.kind === "connect") {
    const several = ask.names.length > 1;
    return (
      <AskFrame
        body={`${joinNames(ask.names)} ${several ? "need" : "needs"} ${ask.service} for their scheduled work. ${several ? "Connect it once and they" : `Connect it and ${ask.names[0]}`} can start.`}
      >
        <Button size="small" onClick={onConnect} disabled={disabled}>
          Connect {ask.service}
        </Button>
        <Button
          size="small"
          variant="ghost"
          className="text-zinc-500"
          onClick={onDismissConnect}
        >
          Not now
        </Button>
      </AskFrame>
    );
  }

  if (ask.kind === "pick") {
    return (
      <AskFrame
        body={
          ask.choices.length > 1
            ? "Which should the team start on first?"
            : "Want to start here?"
        }
        stacked
      >
        {ask.choices.map((choice) => (
          <Button
            key={choice.expertId}
            size="small"
            variant="secondary"
            disabled={disabled}
            onClick={() => onAnswer(choice.message)}
            className="h-auto min-h-9 justify-start whitespace-normal py-2 text-left"
          >
            {choice.label}
          </Button>
        ))}
      </AskFrame>
    );
  }

  return (
    <AskFrame body="Give me a couple of minutes on what a normal week looks like, by voice or text, and I'll work out who and what you need.">
      <Button
        size="small"
        leadingIcon={Mic01Icon}
        onClick={onStartVoice}
        disabled={disabled}
      >
        Talk me through your week
      </Button>
      <Button size="small" variant="secondary" onClick={onType}>
        Type it instead
      </Button>
    </AskFrame>
  );
}

function AskFrame({
  body,
  stacked = false,
  children,
}: {
  body: string;
  stacked?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      aria-label="One thing I need from you"
      className="-mx-5 mt-3 rounded-lg border border-zinc-200 bg-white px-4 py-3"
    >
      <Text variant="small-medium" tone="muted" as="h3">
        One thing I need from you
      </Text>
      <Text variant="body" tone="primary" className="mt-1 text-pretty">
        {body}
      </Text>
      <div
        className={
          stacked
            ? "mt-3 flex flex-col items-stretch gap-2"
            : "mt-3 flex flex-wrap items-center gap-2"
        }
      >
        {children}
      </div>
    </section>
  );
}
