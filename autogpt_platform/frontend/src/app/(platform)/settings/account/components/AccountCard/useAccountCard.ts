"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import type { User } from "@supabase/supabase-js";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { usePostV1UpdateUserEmail } from "@/app/api/__generated__/endpoints/auth/auth";
import { toast } from "@/components/molecules/Toast/use-toast";

const emailSchema = z.object({
  email: z
    .string()
    .min(1, "Email is required")
    .email("Enter a valid email address"),
});

type EmailFormValues = z.infer<typeof emailSchema>;

async function updateEmailViaSupabase(email: string) {
  const response = await fetch("/api/auth/user", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error ?? "Failed to update email");
  }

  return response.json();
}

export function useAccountCard({ user }: { user: User }) {
  const currentEmail = user.email ?? "";

  const emailForm = useForm<EmailFormValues>({
    resolver: zodResolver(emailSchema),
    defaultValues: { email: currentEmail },
    mode: "onChange",
  });

  const updateEmailServer = usePostV1UpdateUserEmail();

  async function onSubmitEmail(values: EmailFormValues): Promise<boolean> {
    if (values.email === currentEmail) return false;

    try {
      await Promise.all([
        updateEmailViaSupabase(values.email),
        updateEmailServer.mutateAsync({ data: values.email }),
      ]);
      toast({
        title: "Email update sent",
        description: "Check your inbox to confirm the change.",
        variant: "success",
      });
      return true;
    } catch (err) {
      toast({
        title: "Couldn't update email",
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      });
      return false;
    }
  }

  return {
    emailForm,
    onSubmitEmail,
    isUpdatingEmail: updateEmailServer.isPending,
    currentEmail,
  };
}
