"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { FormEvent, useState } from "react";

interface Props {
  isEnabling: boolean;
  onEnable: (args: { email: string; note: string }) => Promise<boolean>;
}

export function EnableAccessForm({ isEnabling, onEnable }: Props) {
  const [email, setEmail] = useState("");
  const [note, setNote] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!email.trim()) return;
    if (await onEnable({ email, note })) {
      setEmail("");
      setNote("");
    }
  }

  return (
    <form
      onSubmit={(event) => void handleSubmit(event)}
      className="flex flex-col gap-3 md:flex-row md:items-end"
    >
      <Input
        id="org-access-email"
        label="Account email"
        type="email"
        size="small"
        autoComplete="off"
        spellCheck={false}
        placeholder="name@company.com"
        value={email}
        onChange={(event) => setEmail(event.target.value)}
        disabled={isEnabling}
        wrapperClassName="!mb-0 md:flex-1"
      />
      <Input
        id="org-access-note"
        label="Note"
        size="small"
        placeholder="Team plan trial"
        value={note}
        onChange={(event) => setNote(event.target.value)}
        disabled={isEnabling}
        wrapperClassName="!mb-0 md:flex-1"
      />
      <Button
        type="submit"
        variant="primary"
        size="small"
        loading={isEnabling}
        disabled={!email.trim()}
        className="shrink-0"
      >
        Turn on
      </Button>
    </form>
  );
}
