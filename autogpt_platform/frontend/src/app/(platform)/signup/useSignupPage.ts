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
    setIsGoogleLoading(true);

    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "default",
      });
      setIsGoogleLoading(false);
      resetCaptcha();
      return;
    }
    try {
      const response = await fetch("/api/auth/provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });

      if (!response.ok) {
        const { error } = await response.json();
        setIsGoogleLoading(false);
        resetCaptcha();
        toast({
          title: error || "Failed to start OAuth flow",
          variant: "destructive",
        });
        return;
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
      setFeedback(null);
    } catch (error) {
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
    setIsLoading(true);

    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "default",
      });
      setIsLoading(false);
      resetCaptcha();
      return;
    }

    if (data.email.includes("@agpt.co")) {
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
      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: data.email,
          password: data.password,
          confirmPassword: data.confirmPassword,
          agreeToTerms: data.agreeToTerms,
          turnstileToken: turnstile.token,
        }),
      });

      const result = await response.json();
      setIsLoading(false);

      if (!response.ok) {
        if (result?.error === "user_already_exists") {
          setFeedback("User with this email already exists");
          turnstile.reset();
          return;
        }
        if (result?.error === "not_allowed") {
          setShowNotAllowedModal(true);
          return;
        }
        toast({
          title: result?.error || "Signup failed",
          variant: "destructive",
        });
        resetCaptcha();
        turnstile.reset();
        return;
      }

      setFeedback(null);
      const next = (result?.next as string) || "/";
      router.push(next);
    } catch (error) {
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
