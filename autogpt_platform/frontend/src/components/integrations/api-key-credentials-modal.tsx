import { FC } from "react";
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

export const APIKeyCredentialsModal: FC<{
  schema: BlockIOCredentialsSubSchema;
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
}> = ({ schema, open, onClose, onCredentialsCreate, siblingInputs }) => {
  const credentials = useCredentials(schema, siblingInputs);

  const formSchema = z.object({
    apiKey: z.string().min(1, "API Key is required"),
    title: z.string().min(1, "Name is required"),
    expiresAt: z.string().optional(),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      apiKey: "",
      title: "",
      expiresAt: "",
    },
  });

  if (!credentials || credentials.isLoading || !credentials.supportsApiKey) {
    return null;
  }

  const { provider, providerName, createAPIKeyCredentials } = credentials;

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const expiresAt = values.expiresAt
      ? new Date(values.expiresAt).getTime() / 1000
      : undefined;
    const newCredentials = await createAPIKeyCredentials({
      api_key: values.apiKey,
      title: values.title,
      expires_at: expiresAt,
    });
    onCredentialsCreate({
      provider,
      id: newCredentials.id,
      type: "api_key",
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
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add new API key for {providerName}</DialogTitle>
          {schema.description && (
            <DialogDescription>{schema.description}</DialogDescription>
          )}
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="apiKey"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>API Key</FormLabel>
                  {schema.credentials_scopes && (
                    <FormDescription>
                      Required scope(s) for this block:{" "}
                      {schema.credentials_scopes?.map((s, i, a) => (
                        <span key={i}>
                          <code>{s}</code>
                          {i < a.length - 1 && ", "}
                        </span>
                      ))}
                    </FormDescription>
                  )}
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="Enter API key..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="title"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter a name for this API key..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="expiresAt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Expiration Date (Optional)</FormLabel>
                  <FormControl>
                    <Input
                      type="datetime-local"
                      placeholder="Select expiration date..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit" className="w-full">
              Save & use this API key
            </Button>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
