import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { signup as signupAction } from "./actions";

export function useSignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const { toast } = useToast();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = environment.isCloud();

  const form = useForm<z.infer<typeof signupFormSchema>>({
    resolver: zodResolver(signupFormSchema),
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
      agreeToTerms: false,
    },
  });

  async function handleProviderSignup(provider: LoginProvider) {
    setIsGoogleLoading(true);

    try {
      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });

      if (!response.ok) {
        const { error } = await response.json();

        if (error === "not_allowed") {
          setShowNotAllowedModal(true);
          return;
        }

        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      setIsGoogleLoading(false);
      toast({
        title:
          error instanceof Error ? error.message : "Failed to start OAuth flow",
        variant: "destructive",
      });
    }
  }

  async function handleSignup(data: z.infer<typeof signupFormSchema>) {
    setIsLoading(true);

    if (data.email.includes("@agpt.co")) {
      toast({
        title:
          "Please use Google SSO to create an account using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      return;
    }

    try {
      const result = await signupAction(
        data.email,
        data.password,
        data.confirmPassword,
        data.agreeToTerms,
      );

      setIsLoading(false);

      if (!result.success) {
        if (result.error === "user_already_exists") {
          setFeedback("User with this email already exists");
          return;
        }
        if (result.error === "not_allowed") {
          setShowNotAllowedModal(true);
          return;
        }

        toast({
          title: result.error || "Signup failed",
          variant: "destructive",
        });
        return;
      }

      const next = result.next || "/";
      if (next) router.replace(next);
    } catch (error) {
      setIsLoading(false);
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during signup",
        variant: "destructive",
      });
    }
  }

  return {
    form,
    feedback,
    isLoggedIn: !!user,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isSupabaseAvailable: !!supabase,
    handleSubmit: form.handleSubmit(handleSignup),
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
    handleProviderSignup,
  };
}
