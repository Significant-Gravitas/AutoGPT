"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";

interface Props {
  cardId: string;
  initialMessage: string;
  isStarting: boolean;
  onStart: (message: string) => void;
}

export function DraftStart({
  cardId,
  initialMessage,
  isStarting,
  onStart,
}: Props) {
  const [message, setMessage] = useState(initialMessage);
  const trimmed = message.trim();

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setMessage(event.target.value);
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!trimmed || isStarting) return;
    onStart(trimmed);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="mx-auto flex w-full max-w-2xl flex-col gap-4 p-6"
    >
      <Text variant="body" tone="muted">
        This task hasn&apos;t started yet. Check the first message, then start
        it.
      </Text>
      <Input
        id={`first-message-${cardId}`}
        type="textarea"
        label="First message"
        labelVariant="body-medium"
        rows={6}
        value={message}
        onChange={handleChange}
        wrapperClassName="mb-0"
      />
      <Button
        type="submit"
        size="small"
        leadingIcon={PlayIcon}
        loading={isStarting}
        disabled={!trimmed}
        className="self-end"
      >
        Start
      </Button>
    </form>
  );
}
