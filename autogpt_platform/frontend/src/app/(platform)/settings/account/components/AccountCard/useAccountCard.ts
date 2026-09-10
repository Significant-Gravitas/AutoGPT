"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import type { User } from "@/lib/auth/types";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { useState } from "react";
import { toast } from "@/components/molecules/Toast/use-toast";

const emailSchema = z.object({
  email: z
    .string()
    .min(1, "Email is required")
    .email("Enter a valid email address"),
});

type EmailFormValues = z.infer<typeof emailSchema>;

async function updateEmailViaAuthAPI(email: string) {
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

  const [isUpdatingEmail, setIsUpdatingEmail] = useState(false);

  async function onSubmitEmail(values: EmailFormValues): Promise<boolean> {
    if (values.email === currentEmail) return false;

    // Route the change only through Better Auth. A verified user approves it via
    // a confirmation link sent to their CURRENT address (anti-takeover); an
    // unverified user's change applies immediately. Platform User.email
    // converges via the databaseHooks.user.update mirror in lib/auth/auth.ts —
    // writing it here in parallel would let it diverge to an unverified value.
    // Mirrors EmailForm.
    setIsUpdatingEmail(true);
    try {
      await updateEmailViaAuthAPI(values.email);
      // Reset to the still-current address: on the verified path the row isn't
      // written until the link is clicked, so leaving the new value in the
      // field would let every further click send another email. Mirrors
      // EmailForm.
      emailForm.reset({ email: currentEmail });
      toast(
        user.email_verified
          ? {
              title: "Confirm your new email",
              description:
                "We sent a confirmation link to your current email address. Your email changes once you click it.",
              variant: "success",
            }
          : {
              title: "Email updated",
              description: `Your email is now ${values.email}.`,
              variant: "success",
            },
      );
      return true;
    } catch (err) {
      toast({
        title: "Couldn't update email",
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      });
      return false;
    } finally {
      setIsUpdatingEmail(false);
    }
  }

  return {
    emailForm,
    onSubmitEmail,
    isUpdatingEmail,
    currentEmail,
  };
}
