import { z } from "zod";
import { useForm, type UseFormReturn } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import {
  getGetV1ListCredentialsQueryKey,
  usePostV1CreateCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { APIKeyCredentials } from "@/app/api/__generated__/models/aPIKeyCredentials";
import { useQueryClient } from "@tanstack/react-query";
import { PostV1CreateCredentials201 } from "@/app/api/__generated__/models/postV1CreateCredentials201";

export type APIKeyFormValues = {
  apiKey: string;
  title: string;
  expiresAt?: string;
};

type useAPIKeyCredentialsModalType = {
  schema: BlockIOCredentialsSubSchema;
  onClose: () => void;
  onSuccess: (credentialId: string) => void;
};

export function useAPIKeyCredentialsModal({
  schema,
  onClose,
  onSuccess,
}: useAPIKeyCredentialsModalType): {
  form: UseFormReturn<APIKeyFormValues>;
  isLoading: boolean;
  provider: string;
  schemaDescription?: string;
  onSubmit: (values: APIKeyFormValues) => Promise<void>;
} {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { mutateAsync: createCredentials, isPending: isCreatingCredentials } =
    usePostV1CreateCredentials({
      mutation: {
        onSuccess: async (response) => {
          const credentialId = (response.data as PostV1CreateCredentials201)
            ?.id;
          onClose();
          form.reset();
          toast({
            title: "Success",
            description: "Credentials created successfully",
            variant: "default",
          });

          await queryClient.refetchQueries({
            queryKey: getGetV1ListCredentialsQueryKey(),
          });

          if (credentialId && onSuccess) {
            onSuccess(credentialId);
          }
        },
        onError: () => {
          toast({
            title: "Error",
            description: "Failed to create credentials.",
            variant: "destructive",
          });
        },
      },
    });

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

  async function onSubmit(values: APIKeyFormValues) {
    const expiresAt = values.expiresAt
      ? new Date(values.expiresAt).getTime() / 1000
      : undefined;

    createCredentials({
      provider: schema.credentials_provider[0],
      data: {
        provider: schema.credentials_provider[0],
        type: "api_key",
        api_key: values.apiKey,
        title: values.title,
        expires_at: expiresAt,
      } as APIKeyCredentials,
    });
  }

  return {
    form,
    isLoading: isCreatingCredentials,
    provider: schema.credentials_provider[0],
    schemaDescription: schema.description,
    onSubmit,
  };
}
