import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function useSignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const [captchaKey, setCaptchaKey] = useState(0);
  const { toast } = useToast();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);

  const isCloudEnv = getBehaveAs() === BehaveAs.CLOUD;
  const isVercelPreview = process.env.NEXT_PUBLIC_VERCEL_ENV === "preview";

  const turnstile = useTurnstile({
    action: "signup",
    autoVerify: false,
    resetOnError: true,
  });

  const resetCaptcha = useCallback(() => {
    setCaptchaKey((k) => k + 1);
    turnstile.reset();
  }, [turnstile]);

  const form = useForm<z.infer<typeof signupFormSchema>>({
    resolver: zodResolver(signupFormSchema),
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
      agreeToTerms: false,
    },
  });

  useEffect(() => {
    if (user) router.push("/");
  }, [user]);

  async function handleProviderSignup(provider: LoginProvider) {
    console.log("=== CLIENT OAUTH SIGNUP START ===");
    console.log("Provider:", provider);
    setIsGoogleLoading(true);

    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      console.log("OAuth: CAPTCHA not verified - showing toast");
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "default",
      });
      setIsGoogleLoading(false);
      resetCaptcha();
      return;
    }

    try {
      console.log("Making OAuth provider request...");
      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });

      console.log("OAuth API Response:", {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
      });

      if (!response.ok) {
        const { error } = await response.json();
        console.log("OAuth failed with error:", error);
        setIsGoogleLoading(false);
        resetCaptcha();

        // Check for waitlist error
        if (error === "not_allowed") {
          console.log(">>> OAUTH NOT ALLOWED ERROR DETECTED <<<");
          console.log("Showing not allowed modal");
          setShowNotAllowedModal(true);
          return;
        }

        console.log("Other OAuth error - showing toast:", error);
        toast({
          title: error || "Failed to start OAuth flow",
          variant: "destructive",
        });
        return;
      }

      const { url } = await response.json();
      console.log("OAuth redirect URL received:", url ? "✓" : "✗");
      if (url) {
        console.log("=== CLIENT OAUTH END - REDIRECTING ===");
        window.location.href = url as string;
      }
      setFeedback(null);
    } catch (error) {
      console.error("=== CLIENT OAUTH EXCEPTION ===");
      console.error("Caught error:", error);
      setIsGoogleLoading(false);
      resetCaptcha();
      toast({
        title:
          error instanceof Error ? error.message : "Failed to start OAuth flow",
        variant: "destructive",
      });
    }
  }

  async function handleSignup(data: z.infer<typeof signupFormSchema>) {
    console.log("=== CLIENT SIGNUP START ===");
    console.log("Attempting signup for email:", data.email);
    console.log("Environment:", {
      isCloudEnv,
      isVercelPreview,
      turnstileVerified: turnstile.verified,
      hasTurnstileToken: !!turnstile.token,
    });

    setIsLoading(true);

    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      console.log("CAPTCHA not verified - showing toast");
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "default",
      });
      setIsLoading(false);
      resetCaptcha();
      return;
    }

    if (data.email.includes("@agpt.co")) {
      console.log("AutoGPT email detected - redirecting to Google SSO");
      toast({
        title:
          "Please use Google SSO to create an account using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      resetCaptcha();
      return;
    }

    try {
      console.log("Making signup API request...");
      const requestBody = {
        email: data.email,
        password: data.password,
        confirmPassword: data.confirmPassword,
        agreeToTerms: data.agreeToTerms,
        turnstileToken: turnstile.token,
      };
      console.log("Request body:", {
        ...requestBody,
        password: "[REDACTED]",
        confirmPassword: "[REDACTED]",
      });

      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      console.log("API Response received:", {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
      });

      const result = await response.json();
      console.log("Response body:", result);
      setIsLoading(false);

      if (!response.ok) {
        console.log("Signup failed with error:", result?.error);

        if (result?.error === "user_already_exists") {
          console.log("User already exists error detected");
          setFeedback("User with this email already exists");
          turnstile.reset();
          return;
        }
        if (result?.error === "not_allowed") {
          console.log(">>> NOT ALLOWED ERROR DETECTED <<<");
          console.log("Showing not allowed modal");
          setShowNotAllowedModal(true);
          return;
        }

        console.log("Other signup error - showing toast:", result?.error);
        toast({
          title: result?.error || "Signup failed",
          variant: "destructive",
        });
        resetCaptcha();
        turnstile.reset();
        return;
      }

      console.log("Signup successful!");
      setFeedback(null);
      const next = (result?.next as string) || "/";
      console.log("Redirecting to:", next);
      console.log("=== CLIENT SIGNUP END SUCCESS ===");
      router.push(next);
    } catch (error) {
      console.error("=== CLIENT SIGNUP EXCEPTION ===");
      console.error("Caught error:", error);
      setIsLoading(false);
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during signup",
        variant: "destructive",
      });
      resetCaptcha();
      turnstile.reset();
    }
  }

  return {
    form,
    feedback,
    turnstile,
    captchaKey,
    isLoggedIn: !!user,
    isLoading,
    isCloudEnv,
    isUserLoading,
    isGoogleLoading,
    showNotAllowedModal,
    isSupabaseAvailable: !!supabase,
    handleSubmit: form.handleSubmit(handleSignup),
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
    handleProviderSignup,
  };
}
