import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { login as loginAction } from "./actions";

export function useLoginPage() {
  const { supabase, user, isUserLoading, isLoggedIn } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = environment.isCloud();

  // Get redirect destination from 'next' query parameter
  const nextUrl = searchParams.get("next");

  useEffect(() => {
    if (isLoggedIn && !isLoggingIn) {
      router.push(nextUrl || "/");
    }
  }, [isLoggedIn, isLoggingIn, nextUrl, router]);

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  async function handleProviderLogin(provider: LoginProvider) {
    setIsGoogleLoading(true);
    setIsLoggingIn(true);

    try {
      // Include next URL in OAuth flow if present
      const callbackUrl = nextUrl
        ? `/auth/callback?next=${encodeURIComponent(nextUrl)}`
        : `/auth/callback`;
      const fullCallbackUrl = `${window.location.origin}${callbackUrl}`;

      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, redirectTo: fullCallbackUrl }),
      });

      if (!response.ok) {
        const { error } = await response.json();
        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      setIsGoogleLoading(false);
      setIsLoggingIn(false);
      setFeedback(
        error instanceof Error ? error.message : "Failed to start OAuth flow",
      );
    }
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);
    setIsLoggingIn(true);

    if (data.email.includes("@agpt.co")) {
      toast({
        title: "Please use Google SSO to login using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      setIsLoggingIn(false);
      return;
    }

    try {
      const result = await loginAction(data.email, data.password);

      if (!result.success) {
        throw new Error(result.error || "Login failed");
      }

      // Prefer URL's next parameter, then use backend-determined route
      router.replace(nextUrl || result.next || "/");
    } catch (error) {
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during login",
        variant: "destructive",
      });
      setIsLoading(false);
      setIsLoggingIn(false);
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
