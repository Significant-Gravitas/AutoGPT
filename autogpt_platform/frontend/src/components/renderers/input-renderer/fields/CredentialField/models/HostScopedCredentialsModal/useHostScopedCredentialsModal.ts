import { z } from "zod";
import { useForm, type UseFormReturn } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import {
  getGetV1ListCredentialsQueryKey,
  usePostV1CreateCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { HostScopedCredentialsInput } from "@/app/api/__generated__/models/hostScopedCredentialsInput";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getHostFromUrl } from "@/lib/utils/url";

export type HeaderPair = {
  key: string;
  value: string;
};

export type HostScopedFormValues = {
  host: string;
  title?: string;
};

type UseHostScopedCredentialsModalType = {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
  discriminatorValue?: string;
};

export function useHostScopedCredentialsModal({
  schema,
  provider,
  discriminatorValue,
}: UseHostScopedCredentialsModalType): {
  form: UseFormReturn<HostScopedFormValues>;
  schemaDescription?: string;
  onSubmit: (values: HostScopedFormValues) => Promise<void>;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  headerPairs: HeaderPair[];
  addHeaderPair: () => void;
  removeHeaderPair: (index: number) => void;
  updateHeaderPair: (
    index: number,
    field: "key" | "value",
    value: string,
  ) => void;
  currentHost: string | null;
} {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [headerPairs, setHeaderPairs] = useState<HeaderPair[]>([
    { key: "", value: "" },
  ]);
  const queryClient = useQueryClient();

  // Get current host from discriminatorValue (URL field)
  const currentHost = discriminatorValue
    ? getHostFromUrl(discriminatorValue)
    : null;

  const { mutateAsync: createCredentials } = usePostV1CreateCredentials({
    mutation: {
      onSuccess: async () => {
        form.reset();
        setHeaderPairs([{ key: "", value: "" }]);
        setIsOpen(false);
        toast({
          title: "Success",
          description: "Host-scoped credentials created successfully",
          variant: "default",
        });

        await queryClient.refetchQueries({
          queryKey: getGetV1ListCredentialsQueryKey(),
        });
      },
      onError: () => {
        toast({
          title: "Error",
          description: "Failed to create host-scoped credentials.",
          variant: "destructive",
        });
      },
    },
  });

  const formSchema = z.object({
    host: z.string().min(1, "Host is required"),
    title: z.string().optional().default(""),
  });

  const form = useForm<HostScopedFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      host: currentHost || "",
      title: currentHost || "Manual Entry",
    },
  });

  // Update form values when modal opens and discriminatorValue changes
  const handleSetIsOpen = (open: boolean) => {
    if (open && currentHost) {
      form.setValue("host", currentHost);
      form.setValue("title", currentHost);
    }
    setIsOpen(open);
  };

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

  async function onSubmit(values: HostScopedFormValues) {
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

    createCredentials({
      provider: provider,
      data: {
        provider: provider,
        type: "host_scoped",
        host: values.host,
        title: values.title || values.host,
        headers: headers,
      } as HostScopedCredentialsInput,
    });
  }

  return {
    form,
    schemaDescription: schema.description,
    onSubmit,
    isOpen,
    setIsOpen: handleSetIsOpen,
    headerPairs,
    addHeaderPair,
    removeHeaderPair,
    updateHeaderPair,
    currentHost,
  };
}
