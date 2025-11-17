import { useToast } from "@/components/molecules/Toast/use-toast";
import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { signup as signupAction } from "./actions";

export function useSignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const [captchaKey, setCaptchaKey] = useState(0);
  const { toast } = useToast();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = environment.isCloud();
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
      const result = await signupAction(
        data.email,
        data.password,
        data.confirmPassword,
        data.agreeToTerms,
        turnstile.token ?? undefined,
      );

      setIsLoading(false);

      if (!result.success) {
        if (result.error === "user_already_exists") {
          setFeedback("User with this email already exists");
          turnstile.reset();
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
        resetCaptcha();
        turnstile.reset();
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
