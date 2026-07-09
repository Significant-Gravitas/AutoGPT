import { useToast } from "@/components/molecules/Toast/use-toast";
import { useCaptureMarketingPrompt } from "@/hooks/useCaptureMarketingPrompt";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { signup as signupAction } from "./actions";

export function useSignupPage() {
  useCaptureMarketingPrompt();

  const { supabase, user, isUserLoading, isLoggedIn } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const { toast } = useToast();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [isSigningUp, setIsSigningUp] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = environment.isCloud();

  // Same-origin redirect target; off-site values are dropped so a crafted
  // `/signup?next=https://phishing.site` cannot redirect users elsewhere.
  const nextUrl = sanitizeAuthNext(searchParams.get("next"));

  // Only honour explicit `?next=` deep links here. Generic "already logged in
  // on /signup, get me out" is handled by OnboardingProvider so the user lands
  // straight on /onboarding or /copilot. Otherwise we'd bounce
  // /signup → / → /copilot → /onboarding, and each hop renders before the next
  // redirect — that intermediate /copilot render is the flash users see.
  useEffect(() => {
    if (isLoggedIn && !isSigningUp && nextUrl) {
      router.replace(nextUrl);
    }
  }, [isLoggedIn, isSigningUp, nextUrl, router]);

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
    setIsSigningUp(true);

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

        if (error === "not_allowed") {
          setShowNotAllowedModal(true);
          setIsSigningUp(false);
          return;
        }

        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      setIsGoogleLoading(false);
      setIsSigningUp(false);
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

    setIsSigningUp(true);

    try {
      const result = await signupAction(
        data.email,
        data.password,
        data.confirmPassword,
        data.agreeToTerms,
      );

      if (!result.success) {
        if (result.error === "user_already_exists") {
          setFeedback("User with this email already exists");
          setIsSigningUp(false);
          return;
        }
        if (result.error === "not_allowed") {
          setShowNotAllowedModal(true);
          setIsSigningUp(false);
          return;
        }

        toast({
          title: result.error || "Signup failed",
          variant: "destructive",
        });
        setIsSigningUp(false);
        return;
      }

      // Prefer the URL's next parameter, then result.next (for onboarding), then default
      const redirectTo = nextUrl || result.next || "/";
      router.replace(redirectTo);
    } catch (error) {
      setIsLoading(false);
      setIsSigningUp(false);
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during signup",
        variant: "destructive",
      });
    } finally {
      setTimeout(() => {
        setIsLoading(false);
      }, 3000);
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
