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

  // Get returnUrl, oauth_session, and connect_session from query params
  const returnUrl = searchParams.get("returnUrl");
  const oauthSession = searchParams.get("oauth_session");
  const connectSession = searchParams.get("connect_session");

  function getRedirectUrl(onboarding: boolean): string {
    // OAuth session takes priority - redirect to frontend oauth-resume page
    // which will handle the backend call with proper authentication
    if (oauthSession) {
      return `/auth/oauth-resume?session_id=${encodeURIComponent(oauthSession)}`;
    }

    // Connect session - redirect to frontend connect-resume page
    // for integration credential connection flow
    if (connectSession) {
      return `/auth/connect-resume?session_id=${encodeURIComponent(connectSession)}`;
    }

    // If returnUrl is set and is a valid URL, redirect there after login
    if (returnUrl) {
      try {
        const url = new URL(returnUrl, window.location.origin);
        const backendUrl = process.env.NEXT_PUBLIC_AGPT_SERVER_URL;

        // Same origin - return normalized path only (prevents open redirect)
        if (url.origin === window.location.origin) {
          return url.pathname + url.search;
        }

        // Backend URL - strict origin match (not startsWith to prevent prefix attacks)
        if (backendUrl) {
          try {
            const backendOrigin = new URL(backendUrl).origin;
            if (url.origin === backendOrigin) {
              return url.href;
            }
          } catch {
            // Invalid backend URL config, fall through to default
          }
        }
      } catch {
        // Invalid URL, fall through to default
      }
    }
    return onboarding ? "/onboarding" : "/marketplace";
  }

  useEffect(() => {
    if (isLoggedIn && !isLoggingIn) {
      const redirectTo = getRedirectUrl(false);
      router.push(redirectTo);
    }
  }, [isLoggedIn, isLoggingIn, returnUrl, oauthSession, connectSession]);

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
      // Build redirect URL that preserves oauth_session or connect_session through the OAuth flow
      let callbackUrl: string | undefined;
      const origin = window.location.origin;

      if (oauthSession) {
        callbackUrl = `${origin}/auth/callback?oauth_session=${encodeURIComponent(oauthSession)}`;
      } else if (connectSession) {
        callbackUrl = `${origin}/auth/callback?connect_session=${encodeURIComponent(connectSession)}`;
      }

      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, redirectTo: callbackUrl }),
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

      router.replace(getRedirectUrl(result.onboarding ?? false));
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
