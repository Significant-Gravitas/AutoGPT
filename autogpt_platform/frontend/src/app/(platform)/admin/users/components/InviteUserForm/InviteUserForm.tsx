"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import type { FormEvent } from "react";

interface Props {
  email: string;
  name: string;
  isSubmitting: boolean;
  onEmailChange: (value: string) => void;
  onNameChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}

export function InviteUserForm({
  email,
  name,
  isSubmitting,
  onEmailChange,
  onNameChange,
  onSubmit,
}: Props) {
  return (
    <form className="flex flex-col gap-4" onSubmit={onSubmit}>
      <div className="flex flex-col gap-1">
        <h2 className="text-xl font-semibold text-zinc-900">Create invite</h2>
        <p className="text-sm text-zinc-600">
          The invite is stored immediately, then Tally pre-seeding starts in the
          background.
        </p>
      </div>

      <Input
        id="invite-email"
        label="Email"
        type="email"
        value={email}
        placeholder="jane@example.com"
        autoComplete="email"
        disabled={isSubmitting}
        onChange={(event) => onEmailChange(event.target.value)}
      />

      <Input
        id="invite-name"
        label="Name"
        type="text"
        value={name}
        placeholder="Jane Doe"
        disabled={isSubmitting}
        onChange={(event) => onNameChange(event.target.value)}
      />

      <Button
        type="submit"
        variant="primary"
        loading={isSubmitting}
        disabled={!email.trim()}
        className="w-full"
      >
        {isSubmitting ? "Creating invite..." : "Create invite"}
      </Button>
    </form>
  );
}
