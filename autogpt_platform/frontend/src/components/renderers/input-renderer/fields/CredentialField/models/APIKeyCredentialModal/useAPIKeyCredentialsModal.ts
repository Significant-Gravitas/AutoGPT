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
import { useState } from "react";

export type APIKeyFormValues = {
  apiKey: string;
  title: string;
  expiresAt?: string;
};

type useAPIKeyCredentialsModalType = {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
};

export function useAPIKeyCredentialsModal({
  schema,
  provider,
}: useAPIKeyCredentialsModalType): {
  form: UseFormReturn<APIKeyFormValues>;
  schemaDescription?: string;
  onSubmit: (values: APIKeyFormValues) => Promise<void>;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
} {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const queryClient = useQueryClient();

  const { mutateAsync: createCredentials } = usePostV1CreateCredentials({
    mutation: {
      onSuccess: async () => {
        form.reset();
        setIsOpen(false);
        toast({
          title: "Success",
          description: "Credentials created successfully",
          variant: "default",
        });

        await queryClient.refetchQueries({
          queryKey: getGetV1ListCredentialsQueryKey(),
        });
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
      provider: provider,
      data: {
        provider: provider,
        type: "api_key",
        api_key: values.apiKey,
        title: values.title,
        expires_at: expiresAt,
      } as APIKeyCredentials,
    });
  }

  return {
    form,
    schemaDescription: schema.description,
    onSubmit,
    isOpen,
    setIsOpen,
  };
}
