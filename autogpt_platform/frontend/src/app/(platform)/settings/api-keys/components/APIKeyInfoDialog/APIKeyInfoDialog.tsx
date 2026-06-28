"use client";

import type { ReactNode } from "react";
import { format } from "date-fns";

import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { maskAPIKey } from "../APIKeyList/helpers";
import { humanizePermission } from "../CreateAPIKeyDialog/schema";

interface Props {
  open: boolean;
  apiKey: APIKeyInfo;
  onOpenChange: (open: boolean) => void;
}

export function APIKeyInfoDialog({ open, apiKey, onOpenChange }: Props) {
  return (
    <Dialog
      title={apiKey.name}
      styling={{ maxWidth: "30rem" }}
      controlled={{ isOpen: open, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Section label="Key">
            <code className="font-mono text-sm text-zinc-800">
              {maskAPIKey(apiKey.head, apiKey.tail)}
            </code>
          </Section>

          {apiKey.description && (
            <Section label="Description">
              <Text variant="body" className="text-zinc-700">
                {apiKey.description}
              </Text>
            </Section>
          )}

          <Section label="Scopes">
            {apiKey.scopes.length === 0 ? (
              <Text variant="body" className="text-zinc-500">
                No scopes
              </Text>
            ) : (
              <ul className="flex flex-wrap gap-1.5">
                {apiKey.scopes.map((scope) => (
                  <li
                    key={scope}
                    className="rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-700"
                  >
                    {humanizePermission(scope)}
                  </li>
                ))}
              </ul>
            )}
          </Section>

          <Section label="Created">
            <Text variant="body" className="text-zinc-700">
              {format(new Date(apiKey.created_at), "PPP p")}
            </Text>
          </Section>

          <Section label="Last used">
            <Text variant="body" className="text-zinc-700">
              {apiKey.last_used_at
                ? format(new Date(apiKey.last_used_at), "PPP p")
                : "Never used"}
            </Text>
          </Section>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <Text variant="small-medium" as="span" className="text-zinc-500">
        {label}
      </Text>
      {children}
    </div>
  );
}
