"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { credentialService } from "@/components/contextual/CredentialsInput/service-identity";
import { useId, useState } from "react";
import { useInlineConnectForm } from "./useInlineConnectForm";

interface Props {
  provider: string;
  service: NonNullable<ReturnType<typeof credentialService>>;
  onConnected: (credential?: CredentialsMetaResponse) => void;
}

export function ServiceKeyForm({ provider, service, onConnected }: Props) {
  const keyID = useId();
  const [key, setKey] = useState("");
  const { submit, isPending } = useInlineConnectForm({
    provider,
    successTitle: `${service.name} connection saved`,
    failureTitle: `Couldn't connect ${service.name}`,
    onSuccess: onConnected,
  });

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const token = key.trim();
    if (!token || isPending) return;
    void submit({
      provider,
      type: "host_scoped",
      title: service.name,
      host: service.host,
      headers: { Authorization: service.bearer ? `Bearer ${token}` : token },
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2 rounded-xl bg-surface-muted p-4 text-sm text-foreground-tertiary">
        <p>
          This workflow needs an API key for {service.name}. Otto will use it
          only for requests to {service.host}.
        </p>
        <p>{service.keyHelp}</p>
      </div>
      <Input
        id={keyID}
        label="API key"
        type="password"
        autoComplete="new-password"
        spellCheck={false}
        value={key}
        onChange={(event) => setKey(event.target.value)}
      />
      <Button
        type="submit"
        variant="primary"
        size="small"
        className="w-full"
        loading={isPending}
        disabled={!key.trim() || isPending}
      >
        {isPending ? "Connecting…" : "Save and connect"}
      </Button>
    </form>
  );
}
