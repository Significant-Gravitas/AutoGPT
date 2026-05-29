"use client";

import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { User } from "@supabase/supabase-js";
import { usePostV1UpdateUserEmail } from "@/app/api/__generated__/endpoints/auth/auth";

const emailFormSchema = z.object({
  email: z
    .string()
    .min(1, "Email is required")
    .email("Please enter a valid email address"),
});

function createEmailDefaultValues(user: { email?: string }) {
  return {
    email: user.email || "",
  };
}

async function updateUserEmailAPI(email: string) {
  const response = await fetch("/api/auth/user", {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Failed to update email");
  }

  return response.json();
}

export function useEmailForm({ user }: { user: User }) {
  const { toast } = useToast();
  const defaultValues = createEmailDefaultValues(user);
  const currentEmail = user.email;

  const form = useForm<z.infer<typeof emailFormSchema>>({
    resolver: zodResolver(emailFormSchema),
    defaultValues,
    mode: "onSubmit",
  });

  const updateEmailMutation = usePostV1UpdateUserEmail({
    mutation: {
      onError: (error) => {
        toast({
          title: "Error updating email",
          description:
            error instanceof Error ? error.message : "Failed to update email",
          variant: "destructive",
        });
      },
    },
  });

  async function onSubmit(values: z.infer<typeof emailFormSchema>) {
    try {
      if (values.email !== user.email) {
        await Promise.all([
          updateUserEmailAPI(values.email),
          updateEmailMutation.mutateAsync({ data: values.email }),
        ]);

        toast({
          title: "Successfully updated email",
        });
      }
    } catch (error) {
      toast({
        title: "Error updating email",
        description:
          error instanceof Error ? error.message : "Something went wrong",
        variant: "destructive",
      });
    }
  }

  return {
    form,
    onSubmit,
    isLoading: updateEmailMutation.isPending,
    currentEmail,
  };
}
