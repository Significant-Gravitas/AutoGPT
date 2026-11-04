"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { FormEvent, useState } from "react";

interface Props {
  initialEmail: string;
  isLoading: boolean;
  onLookUp: (email: string) => void;
}

export function AccountLookupForm({
  initialEmail,
  isLoading,
  onLookUp,
}: Props) {
  const [email, setEmail] = useState(initialEmail);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (email.trim()) onLookUp(email);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-3 md:flex-row md:items-end"
    >
      <Input
        id="team-plans-email"
        label="Owner or member email"
        type="email"
        size="small"
        autoComplete="off"
        spellCheck={false}
        placeholder="name@company.com"
        value={email}
        onChange={(event) => setEmail(event.target.value)}
        wrapperClassName="!mb-0 md:flex-1"
      />
      <Button
        type="submit"
        variant="primary"
        size="small"
        loading={isLoading}
        disabled={!email.trim()}
        className="shrink-0"
      >
        Look up
      </Button>
    </form>
  );
}
