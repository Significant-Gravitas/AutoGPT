"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/atoms/Table/Table";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";
import { DeleteProviderModal } from "./DeleteProviderModal";
import { EditProviderModal } from "./EditProviderModal";

export function ProviderList({ providers }: { providers: LlmProvider[] }) {
  if (!providers.length) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        No providers configured yet.
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Display Name</TableHead>
            <TableHead>Default Credential</TableHead>
            <TableHead>Capabilities</TableHead>
            <TableHead>Models</TableHead>
            <TableHead className="w-[100px]">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {providers.map((provider) => (
            <TableRow key={provider.id}>
              <TableCell className="font-medium">{provider.name}</TableCell>
              <TableCell>{provider.display_name}</TableCell>
              <TableCell>
                {provider.default_credential_provider
                  ? `${provider.default_credential_provider} (${provider.default_credential_id ?? "id?"})`
                  : "—"}
              </TableCell>
              <TableCell className="text-sm text-muted-foreground">
                <div className="flex flex-wrap gap-2">
                  {provider.supports_tools && (
                    <span className="rounded bg-muted px-2 py-0.5 text-xs">
                      Tools
                    </span>
                  )}
                  {provider.supports_json_output && (
                    <span className="rounded bg-muted px-2 py-0.5 text-xs">
                      JSON
                    </span>
                  )}
                  {provider.supports_reasoning && (
                    <span className="rounded bg-muted px-2 py-0.5 text-xs">
                      Reasoning
                    </span>
                  )}
                  {provider.supports_parallel_tool && (
                    <span className="rounded bg-muted px-2 py-0.5 text-xs">
                      Parallel Tools
                    </span>
                  )}
                </div>
              </TableCell>
              <TableCell className="text-sm">
                <span
                  className={
                    (provider.models?.length ?? 0) > 0
                      ? "text-foreground"
                      : "text-muted-foreground"
                  }
                >
                  {provider.models?.length ?? 0}
                </span>
              </TableCell>
              <TableCell>
                <div className="flex gap-2">
                  <EditProviderModal provider={provider} />
                  <DeleteProviderModal provider={provider} />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
