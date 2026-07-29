import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { normalizeMCPUrl } from "@/lib/utils/url";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm, type UseFormReturn } from "react-hook-form";
import { z } from "zod";

export type APIKeyFormValues = {
  apiKey: string;
  title: string;
  expiresAt?: string;
};

type Args = {
  schema: BlockIOCredentialsSubSchema;
  siblingInputs?: Record<string, any>;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
};

export function useAPIKeyCredentialsModal({
  schema,
  siblingInputs,
  onCredentialsCreate,
}: Args): {
  form: UseFormReturn<APIKeyFormValues>;
  isLoading: boolean;
  isSubmitting: boolean;
  supportsApiKey: boolean;
  provider?: string;
  providerName?: string;
  schemaDescription?: string;
  onSubmit: (values: APIKeyFormValues) => Promise<void>;
} {
  const credentials = useCredentials(schema, siblingInputs);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const formSchema = z.object({
    apiKey: z.string().min(1, "API Key is required"),
    title: z.string().min(1, "Name is required"),
    expiresAt: z.string().optional(),
  });

  const form = useForm<APIKeyFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      apiKey: "",
      title: "",
      expiresAt: "",
    },
  });

  const mcpServerUrl =
    credentials && !credentials.isLoading && credentials.provider === "mcp"
      ? normalizeMCPUrl(credentials.discriminatorValue ?? "")
      : "";

  async function onSubmit(values: APIKeyFormValues) {
    if (!credentials || credentials.isLoading) return;
    setIsSubmitting(true);
    try {
      const expiresAt = values.expiresAt
        ? new Date(values.expiresAt).getTime() / 1000
        : undefined;
      const newCredentials = await credentials.createAPIKeyCredentials({
        api_key: values.apiKey,
        title: values.title,
        expires_at: expiresAt,
        // MCP credentials are matched to a node by the server URL held in
        // `metadata.mcp_server_url`. Without it the new credential comes back
        // with `host: null`, gets filtered out of the picker, and the
        // selection made here is immediately cleared again.
        ...(mcpServerUrl && { metadata: { mcp_server_url: mcpServerUrl } }),
      });
      onCredentialsCreate({
        provider: credentials.provider,
        id: newCredentials.id,
        type: "api_key",
        title: newCredentials.title,
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    form,
    isLoading: !credentials || credentials.isLoading,
    isSubmitting,
    supportsApiKey: !!credentials?.supportsApiKey,
    provider: credentials?.provider,
    providerName:
      !credentials || credentials.isLoading
        ? undefined
        : credentials.providerName,
    schemaDescription: schema.description,
    onSubmit,
  };
}
