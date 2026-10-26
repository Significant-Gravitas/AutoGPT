import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { expiryInputToUnix } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/schema";
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
    apiKey: z.string().trim().min(1, "API Key is required"),
    title: z.string().trim().min(1, "Name is required"),
    expiresAt: z
      .string()
      .optional()
      .superRefine((value, ctx) => {
        if (!value) return;
        const expiresAt = expiryInputToUnix(value);
        if (expiresAt === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Pick a valid date",
          });
        } else if (expiresAt * 1000 <= Date.now()) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Expiry must be in the future",
          });
        }
      }),
  });

  const form = useForm<APIKeyFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      apiKey: "",
      title: "",
      expiresAt: "",
    },
  });

  async function onSubmit(values: APIKeyFormValues) {
    if (!credentials || credentials.isLoading) return;
    setIsSubmitting(true);
    try {
      const expiresAt = values.expiresAt
        ? expiryInputToUnix(values.expiresAt)
        : undefined;
      const newCredentials = await credentials.createAPIKeyCredentials({
        api_key: values.apiKey.trim(),
        title: values.title.trim(),
        expires_at: expiresAt,
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
