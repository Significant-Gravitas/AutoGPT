import useCredentials from "@/hooks/useCredentials";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { zodResolver } from "@hookform/resolvers/zod";
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
  supportsApiKey: boolean;
  provider?: string;
  providerName?: string;
  schemaDescription?: string;
  onSubmit: (values: APIKeyFormValues) => Promise<void>;
} {
  const credentials = useCredentials(schema, siblingInputs);

  const formSchema = z.object({
    apiKey: z.string().min(1, "API Key is required"),
    title: z.string().min(1, "Name is required"),
    expiresAt: z.string().optional(),
  });

  function getDefaultExpirationDate(): string {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(0, 0, 0, 0);
    const year = tomorrow.getFullYear();
    const month = String(tomorrow.getMonth() + 1).padStart(2, "0");
    const day = String(tomorrow.getDate()).padStart(2, "0");
    const hours = String(tomorrow.getHours()).padStart(2, "0");
    const minutes = String(tomorrow.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day}T${hours}:${minutes}`;
  }

  const form = useForm<APIKeyFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      apiKey: "",
      title: "",
      expiresAt: getDefaultExpirationDate(),
    },
  });

  async function onSubmit(values: APIKeyFormValues) {
    if (!credentials || credentials.isLoading) return;
    const expiresAt = values.expiresAt
      ? new Date(values.expiresAt).getTime() / 1000
      : undefined;
    const newCredentials = await credentials.createAPIKeyCredentials({
      api_key: values.apiKey,
      title: values.title,
      expires_at: expiresAt,
    });
    onCredentialsCreate({
      provider: credentials.provider,
      id: newCredentials.id,
      type: "api_key",
      title: newCredentials.title,
    });
  }

  return {
    form,
    isLoading: !credentials || credentials.isLoading,
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
