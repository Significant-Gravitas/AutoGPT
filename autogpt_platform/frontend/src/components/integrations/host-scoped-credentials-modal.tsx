import { FC, useEffect, useState } from "react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { getHostFromUrl } from "@/lib/utils/url";

export const HostScopedCredentialsModal: FC<{
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
}> = ({ schema, open, onClose, onCredentialsCreate, siblingInputs }) => {
  const credentials = useCredentials(schema, siblingInputs);

  // Get current host from siblingInputs or discriminator_values
  const currentUrl = credentials?.discriminatorValue;
  const currentHost = currentUrl ? getHostFromUrl(currentUrl) : "";

  const formSchema = z.object({
    host: z.string().min(1, "Host is required"),
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

  const [headerPairs, setHeaderPairs] = useState<
    Array<{ key: string; value: string }>
  >([{ key: "", value: "" }]);

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

  const { provider, providerName, createHostScopedCredentials } = credentials;

  const addHeaderPair = () => {
    setHeaderPairs([...headerPairs, { key: "", value: "" }]);
  };

  const removeHeaderPair = (index: number) => {
    if (headerPairs.length > 1) {
      setHeaderPairs(headerPairs.filter((_, i) => i !== index));
    }
  };

  const updateHeaderPair = (
    index: number,
    field: "key" | "value",
    value: string,
  ) => {
    const newPairs = [...headerPairs];
    newPairs[index][field] = value;
    setHeaderPairs(newPairs);
  };

  async function onSubmit(values: z.infer<typeof formSchema>) {
    // Convert header pairs to object, filtering out empty pairs
    const headers = headerPairs.reduce(
      (acc, pair) => {
        if (pair.key.trim() && pair.value.trim()) {
          acc[pair.key.trim()] = pair.value.trim();
        }
        return acc;
      },
      {} as Record<string, string>,
    );

    const newCredentials = await createHostScopedCredentials({
      host: values.host,
      title: currentHost || values.host,
      headers,
    });

    onCredentialsCreate({
      provider,
      id: newCredentials.id,
      type: "host_scoped",
      title: newCredentials.title,
    });
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add sensitive headers for {providerName}</DialogTitle>
          {schema.description && (
            <DialogDescription>{schema.description}</DialogDescription>
          )}
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="host"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Host Pattern</FormLabel>
                  <FormDescription>
                    {currentHost
                      ? "Auto-populated from the URL field. Headers will be applied to requests to this host."
                      : "Enter the host/domain to match against request URLs (e.g., api.example.com)."}
                  </FormDescription>
                  <FormControl>
                    <Input
                      type="text"
                      readOnly={!!currentHost}
                      placeholder={
                        currentHost
                          ? undefined
                          : "Enter host (e.g., api.example.com)"
                      }
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="space-y-2">
              <FormLabel>Headers</FormLabel>
              <FormDescription>
                Add sensitive headers (like Authorization, X-API-Key) that
                should be automatically included in requests to the specified
                host.
              </FormDescription>

              {headerPairs.map((pair, index) => (
                <div key={index} className="flex items-end gap-2">
                  <div className="flex-1">
                    <Input
                      placeholder="Header name (e.g., Authorization)"
                      value={pair.key}
                      onChange={(e) =>
                        updateHeaderPair(index, "key", e.target.value)
                      }
                    />
                  </div>
                  <div className="flex-1">
                    <Input
                      type="password"
                      placeholder="Header value (e.g., Bearer token123)"
                      value={pair.value}
                      onChange={(e) =>
                        updateHeaderPair(index, "value", e.target.value)
                      }
                    />
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => removeHeaderPair(index)}
                    disabled={headerPairs.length === 1}
                  >
                    Remove
                  </Button>
                </div>
              ))}

              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addHeaderPair}
                className="w-full"
              >
                Add Another Header
              </Button>
            </div>

            <Button type="submit" className="w-full">
              Save & use these credentials
            </Button>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
