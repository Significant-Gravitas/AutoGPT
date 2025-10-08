import { useState } from "react";
import z from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  getGetV1ListCredentialsQueryKey,
  usePostV1CreateCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

type usePasswordCredentialModalType = {
  provider: string;
};

export const usePasswordCredentialModal = ({
  provider,
}: usePasswordCredentialModalType) => {
  const [open, setOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const formSchema = z.object({
    username: z.string().min(1, "Username is required"),
    password: z.string().min(1, "Password is required"),
    title: z.string().min(1, "Name is required"),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
      password: "",
      title: "",
    },
  });

  const { mutateAsync: createCredentials } = usePostV1CreateCredentials({
    mutation: {
      onSuccess: async () => {
        form.reset();
        setOpen(false);
        toast({
          title: "Success",
          description: "Credentials created successfully",
          variant: "default",
        });

        await queryClient.refetchQueries({
          queryKey: getGetV1ListCredentialsQueryKey(),
        });
      },
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    createCredentials({
      provider: provider,
      data: {
        provider: provider,
        type: "user_password",
        username: values.username,
        password: values.password,
        title: values.title,
      },
    });
  }

  return {
    form,
    onSubmit,
    open,
    setOpen,
  };
};
