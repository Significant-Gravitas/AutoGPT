import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { login as loginAction } from "./actions";

export function useLoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = environment.isCloud();

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  async function handleProviderLogin(provider: LoginProvider) {
    setIsGoogleLoading(true);

    try {
      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });

      if (!response.ok) {
        const { error } = await response.json();
        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      setIsGoogleLoading(false);
      setFeedback(
        error instanceof Error ? error.message : "Failed to start OAuth flow",
      );
    }
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);

    if (data.email.includes("@agpt.co")) {
      toast({
        title: "Please use Google SSO to login using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      return;
    }

    try {
      const result = await loginAction(data.email, data.password);

      if (!result.success) {
        throw new Error(result.error || "Login failed");
      }

      if (result.onboarding) {
        router.replace("/onboarding");
      } else {
        router.replace("/marketplace");
      }
    } catch (error) {
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during login",
        variant: "destructive",
      });
      setIsLoading(false);
    }
  }

  return {
    form,
    feedback,
    user,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isSupabaseAvailable: !!supabase,
    handleSubmit: form.handleSubmit(handleLogin),
    handleProviderLogin,
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
  };
}
