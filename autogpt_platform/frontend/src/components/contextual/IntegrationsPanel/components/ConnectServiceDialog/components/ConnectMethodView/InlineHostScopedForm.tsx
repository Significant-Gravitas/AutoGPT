"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Delete02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import {
  addHeaderPairToList,
  headerPairsToRecord,
  removeHeaderPairFromList,
  updateHeaderPairInList,
  type HeaderPair,
} from "@/components/contextual/CredentialsInput/helpers";

import { useInlineConnectForm } from "./useInlineConnectForm";

const hostScopedConnectSchema = z
  .object({
    host: z
      .string()
      .trim()
      .min(1, "Host is required")
      .refine((value) => !/^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//.test(value), {
        message: "Enter only the host (e.g. api.example.com), not a full URL",
      })
      .refine((value) => !value.includes("/"), {
        message:
          "Enter only the host, without a trailing path. A port is allowed.",
      }),
  })
  .strict();

type HostScopedConnectFormValues = z.infer<typeof hostScopedConnectSchema>;

interface Props {
  provider: string;
  /** The host the requesting block will call. Absent where no block is in
   *  scope — the settings dialog and the copilot connect card. */
  host?: string;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

// Host pattern plus the headers to attach to requests matching it. Carries its
// own submit, because the panel footer's Continue only drives OAuth and API key.
export function InlineHostScopedForm({ provider, host, onSuccess }: Props) {
  const [headerPairs, setHeaderPairs] = useState<HeaderPair[]>([
    { key: "", value: "" },
  ]);
  const { submit, isPending } = useInlineConnectForm({
    provider,
    successTitle: "Host credentials saved",
    failureTitle: "Couldn't save host credentials",
    onSuccess,
  });

  const form = useForm<HostScopedConnectFormValues>({
    resolver: zodResolver(hostScopedConnectSchema),
    defaultValues: { host: host ?? "" },
    mode: "onChange",
  });

  const headers = headerPairsToRecord(headerPairs);
  // Parsed rather than read off formState, which stays invalid until the
  // first change and so would reject a host we prefilled.
  const hostIsValid = hostScopedConnectSchema.safeParse({
    host: form.watch("host"),
  }).success;
  // A host-scoped credential with no headers adds nothing to a request, so
  // saving one would silently produce a credential that does nothing.
  const canSubmit = hostIsValid && Object.keys(headers).length > 0;

  function handleSubmit(values: HostScopedConnectFormValues) {
    submit({
      provider,
      type: "host_scoped",
      title: values.host,
      host: values.host,
      headers,
    });
  }

  return (
    <Form form={form} onSubmit={handleSubmit} className="space-y-2.5">
      <FormField
        control={form.control}
        name="host"
        render={({ field }) => (
          <FormItem>
            <FormControl>
              <Input
                {...field}
                id="host"
                autoComplete="off"
                spellCheck={false}
                label="Host"
                labelVariant="small-medium"
                size="small"
                readOnly={Boolean(host)}
                placeholder="api.example.com"
                wrapperClassName="!mb-0"
              />
            </FormControl>
            <Text variant="small" className="!text-zinc-500">
              {host
                ? "Taken from the URL this block calls."
                : "The host of the URL this block will call."}
            </Text>
            <FormMessage />
          </FormItem>
        )}
      />

      <div className="space-y-2">
        <Text variant="small-medium" className="text-zinc-900">
          Headers
        </Text>
        <Text variant="small" className="!text-zinc-500">
          Sent with every request to this host, such as Authorization.
        </Text>

        {headerPairs.map((pair, index) => (
          <div key={index} className="flex items-end gap-2">
            <Input
              id={`header-${index}-key`}
              label="Header name"
              hideLabel
              size="small"
              className="flex-1"
              wrapperClassName="!mb-0 flex-1"
              placeholder="Authorization"
              value={pair.key}
              onChange={(event) =>
                updateHeaderPair(index, "key", event.target.value)
              }
            />
            <Input
              id={`header-${index}-value`}
              label="Header value"
              hideLabel
              type="password"
              autoComplete="new-password"
              size="small"
              className="flex-1"
              wrapperClassName="!mb-0 flex-1"
              placeholder="Bearer …"
              value={pair.value}
              onChange={(event) =>
                updateHeaderPair(index, "value", event.target.value)
              }
            />
            <Button
              type="button"
              variant="icon"
              size="icon"
              aria-label={`Remove header ${index + 1}`}
              disabled={headerPairs.length === 1}
              onClick={() => removeHeaderPair(index)}
            >
              <Icon icon={Delete02Icon} size={16} />
            </Button>
          </div>
        ))}

        <Button
          type="button"
          variant="outline"
          size="small"
          onClick={addHeaderPair}
        >
          <Icon icon={PlusSignIcon} size={16} /> Add header
        </Button>
      </div>

      <Button
        type="submit"
        variant="primary"
        size="small"
        className="w-full"
        disabled={!canSubmit}
        loading={isPending}
      >
        {isPending ? "Connecting…" : "Connect"}
      </Button>
    </Form>
  );

  function addHeaderPair() {
    setHeaderPairs((pairs) => addHeaderPairToList(pairs));
  }

  function removeHeaderPair(index: number) {
    setHeaderPairs((pairs) => removeHeaderPairFromList(pairs, index));
  }

  function updateHeaderPair(
    index: number,
    field: "key" | "value",
    value: string,
  ) {
    setHeaderPairs((pairs) =>
      updateHeaderPairInList(pairs, index, field, value),
    );
  }
}
