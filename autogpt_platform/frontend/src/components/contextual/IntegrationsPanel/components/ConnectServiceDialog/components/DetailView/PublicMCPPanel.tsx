"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useCheckMCPConnection } from "./useCheckMCPConnection";

interface Props {
  serverURL: string;
}

export function PublicMCPPanel({ serverURL }: Props) {
  const { check, isPending, toolCount, error } =
    useCheckMCPConnection(serverURL);
  return (
    <div className="flex flex-col gap-4">
      <Input
        id="public-mcp-server-url"
        label="Server URL"
        type="url"
        value={serverURL}
        readOnly
      />
      <Text variant="small" className="text-zinc-600">
        This server can be used without connecting an account.
      </Text>
      {toolCount !== null && (
        <Text variant="body" role="status">
          {toolCount === 0
            ? "Connected, but this server returned no tools."
            : `${toolCount} tools available. This server is ready to use in your agents.`}
        </Text>
      )}
      {error && (
        <Text variant="small" role="alert" className="text-red-700">
          {error}
        </Text>
      )}
      <Button
        variant="primary"
        size="small"
        onClick={check}
        loading={isPending}
      >
        Check connection
      </Button>
    </div>
  );
}
