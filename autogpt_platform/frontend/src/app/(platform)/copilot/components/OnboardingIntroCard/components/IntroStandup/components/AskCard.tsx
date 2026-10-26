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
  onStartVoice: () => void;
  onType: () => void;
  canRecord: boolean;
}

export function AskCard({
  ask,
  disabled,
  onConnect,
  onDismissConnect,
  onStartVoice,
  onType,
  canRecord,
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
          className="text-foreground-tertiary"
          onClick={onDismissConnect}
        >
          Not now
        </Button>
      </AskFrame>
    );
  }

  if (!canRecord) {
    return (
      <AskFrame body="Give me a couple of minutes on what a normal week looks like, and I'll work out who and what you need.">
        <Button size="small" onClick={onType}>
          Type it instead
        </Button>
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

function AskFrame({ body, children }: { body: string; children: ReactNode }) {
  return (
    <section
      aria-label="One thing I need from you"
      className="mt-3 rounded-lg border border-border-base bg-surface px-4 py-3 md:-mx-5"
    >
      <Text variant="small-medium" tone="muted" as="h3">
        One thing I need from you
      </Text>
      <Text variant="body" tone="primary" className="mt-1 text-pretty">
        {body}
      </Text>
      <div className="mt-3 flex flex-wrap items-center gap-2">{children}</div>
    </section>
  );
}
