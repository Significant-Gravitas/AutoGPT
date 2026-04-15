import { useContext, useEffect, useState } from "react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormDescription,
  FormField,
  FormLabel,
} from "@/components/__legacy__/ui/form";
import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { getHostFromUrl } from "@/lib/utils/url";
import { PlusIcon, TrashIcon } from "@phosphor-icons/react";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  addHeaderPairToList,
  findExistingHostCredentials,
  hasExistingHostCredential,
  headerPairsToRecord,
  removeHeaderPairFromList,
  updateHeaderPairInList,
  type HeaderPair,
} from "../../helpers";

type Props = {
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
};

export function HostScopedCredentialsModal({
  schema,
  open,
  onClose,
  onCredentialsCreate,
  siblingInputs,
}: Props) {
  const credentials = useCredentials(schema, siblingInputs);
  const allProviders = useContext(CredentialsProvidersContext);

  // Get current host from siblingInputs or discriminator_values
  const currentUrl = credentials?.discriminatorValue;
  const currentHost = currentUrl ? getHostFromUrl(currentUrl) : "";

  const formSchema = z.object({
    host: z
      .string()
      .min(1, "Host is required")
      .refine((val) => !/^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//.test(val), {
        message: "Enter only the host (e.g. api.example.com), not a full URL",
      })
      .refine((val) => !val.includes("/"), {
        message:
          "Enter only the host (e.g. api.example.com), without a trailing path. " +
          "You may specify a port (e.g. api.example.com:8080) if needed.",
      }),
    title: z.string().optional(),
    headers: z.record(z.string()).optional(),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      host: currentHost || "",
      title: currentHost || "Manual Entry",
      headers: {},
    },
  });

  const [headerPairs, setHeaderPairs] = useState<HeaderPair[]>([
    { key: "", value: "" },
  ]);

  // Update form values when siblingInputs change
  useEffect(() => {
    if (currentHost) {
      form.setValue("host", currentHost);
      form.setValue("title", currentHost);
    } else {
      // Reset to empty when no current host
      form.setValue("host", "");
      form.setValue("title", "Manual Entry");
    }
  }, [currentHost, form]);

  if (
    !credentials ||
    credentials.isLoading ||
    !credentials.supportsHostScoped
  ) {
    return null;
  }

  const {
    provider,
    providerName,
    createHostScopedCredentials,
    deleteCredentials,
  } = credentials;

  // Use the unfiltered credential list from the provider context for deduplication.
  // The hook's savedCredentials is pre-filtered by discriminatorValue, which may be
  // empty when no URL is entered yet — causing deduplication to miss existing creds.
  const allProviderCredentials =
    allProviders?.[provider]?.savedCredentials ?? [];

  const hasExistingForHost = hasExistingHostCredential(
    allProviderCredentials,
    currentHost || form.getValues("host"),
  );

  const addHeaderPair = () => {
    setHeaderPairs((prev) => addHeaderPairToList(prev));
  };

  const removeHeaderPair = (index: number) => {
    setHeaderPairs((prev) => removeHeaderPairFromList(prev, index));
  };

  const updateHeaderPair = (
    index: number,
    field: "key" | "value",
    value: string,
  ) => {
    setHeaderPairs((prev) => updateHeaderPairInList(prev, index, field, value));
  };

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const headers = headerPairsToRecord(headerPairs);

    // Delete existing host-scoped credentials for the same host to avoid duplicates.
    // Uses unfiltered provider credentials (not the hook's pre-filtered list).
    const host = values.host;
    const existingForHost = findExistingHostCredentials(
      allProviderCredentials,
      host,
    );

    try {
      for (const existing of existingForHost) {
        await deleteCredentials(existing.id, true);
      }

      const newCredentials = await createHostScopedCredentials({
        host,
        title: currentHost || host,
        headers,
      });

      onCredentialsCreate({
        provider,
        id: newCredentials.id,
        type: "host_scoped",
        title: newCredentials.title,
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong";
      toast({
        title: "Failed to save credentials",
        description: message,
        variant: "destructive",
      });
    }
  }

  return (
    <Dialog
      title={
        hasExistingForHost
          ? `Update sensitive headers for ${providerName}`
          : `Add sensitive headers for ${providerName}`
      }
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
      onClose={onClose}
      styling={{
        maxWidth: "25rem",
      }}
    >
      <Dialog.Content>
        {schema.description && (
          <p className="mb-4 text-sm text-zinc-600">{schema.description}</p>
        )}

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-2">
            <FormField
              control={form.control}
              name="host"
              render={({ field }) => (
                <Input
                  id="host"
                  label="Host Pattern"
                  type="text"
                  size="small"
                  readOnly={!!currentHost}
                  hint={
                    currentHost
                      ? "Auto-populated from the URL field. Headers will be applied to requests to this host."
                      : "Enter the host/domain to match against request URLs (e.g., api.example.com)."
                  }
                  placeholder={
                    currentHost
                      ? undefined
                      : "Enter host (e.g., api.example.com)"
                  }
                  {...field}
                />
              )}
            />

            <div className="space-y-2">
              <FormLabel>Headers</FormLabel>
              <FormDescription className="max-w-md">
                Add sensitive headers (like Authorization, X-API-Key) that
                should be automatically included in requests to the specified
                host.
              </FormDescription>

              {headerPairs.map((pair, index) => (
                <div key={index} className="flex w-full items-center gap-4">
                  <Input
                    id={`header-${index}-key`}
                    label="Header Name"
                    placeholder="Header name (e.g., Authorization)"
                    size="small"
                    value={pair.key}
                    className="flex-1"
                    onChange={(e) =>
                      updateHeaderPair(index, "key", e.target.value)
                    }
                  />

                  <Input
                    id={`header-${index}-value`}
                    label="Header Value"
                    size="small"
                    type="password"
                    className="flex-2"
                    placeholder="Header value (e.g., Bearer token123)"
                    value={pair.value}
                    onChange={(e) =>
                      updateHeaderPair(index, "value", e.target.value)
                    }
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="small"
                    onClick={() => removeHeaderPair(index)}
                    disabled={headerPairs.length === 1}
                  >
                    <TrashIcon className="size-4" /> Remove
                  </Button>
                </div>
              ))}

              <Button
                type="button"
                variant="outline"
                size="small"
                onClick={addHeaderPair}
              >
                <PlusIcon className="size-4" /> Add Another Header
              </Button>
            </div>

            <div className="pt-8">
              <Button type="submit" className="w-full" size="small">
                {hasExistingForHost
                  ? "Update & use these credentials"
                  : "Save & use these credentials"}
              </Button>
            </div>
          </form>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
