"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { User } from "@/lib/auth/types";

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

// Better Auth owns the email change. For a verified user it emails a
// confirmation link to their CURRENT address and only applies the new email
// once that link is clicked (anti-takeover); for an unverified user the change
// applies immediately. Platform User.email (notifications / Stripe) then
// converges via the databaseHooks.user.update mirror in lib/auth/auth.ts — we
// deliberately do NOT write the platform email here, so it can never diverge to
// an unverified value.
async function requestEmailChange(email: string) {
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
  const [isSubmitting, setIsSubmitting] = useState(false);

  const form = useForm<z.infer<typeof emailFormSchema>>({
    resolver: zodResolver(emailFormSchema),
    defaultValues,
    mode: "onSubmit",
  });

  async function onSubmit(values: z.infer<typeof emailFormSchema>) {
    if (values.email === user.email) return;

    setIsSubmitting(true);
    try {
      await requestEmailChange(values.email);
      // Reset to the still-current address. On the verified path Better Auth
      // deliberately doesn't write the row until the link is clicked, so
      // `user.email` stays old — leaving the new value in the field would keep
      // the submit button enabled and fire another email on every click.
      form.reset({ email: currentEmail });
      toast(
        user.email_verified
          ? {
              title: "Confirm your new email",
              description:
                "We sent a confirmation link to your current email address. Your email changes once you click it.",
            }
          : {
              title: "Email updated",
              description: `Your email is now ${values.email}.`,
            },
      );
    } catch (error) {
      toast({
        title: "Error updating email",
        description:
          error instanceof Error ? error.message : "Something went wrong",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    form,
    onSubmit,
    isLoading: isSubmitting,
    currentEmail,
  };
}
