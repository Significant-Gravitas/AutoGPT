import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function useLoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const [captchaKey, setCaptchaKey] = useState(0);
  const router = useRouter();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = getBehaveAs() === BehaveAs.CLOUD;
  const isVercelPreview = process.env.NEXT_PUBLIC_VERCEL_ENV === "preview";

  const turnstile = useTurnstile({
    action: "login",
    autoVerify: false,
    resetOnError: true,
  });

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const resetCaptcha = useCallback(() => {
    setCaptchaKey((k) => k + 1);
    turnstile.reset();
  }, [turnstile]);

  useEffect(() => {
    if (user) router.push("/");
  }, [user]);

  async function handleProviderLogin(provider: LoginProvider) {
    setIsGoogleLoading(true);

    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "info",
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
        if (typeof error === "string" && error.includes("not_allowed")) {
          setShowNotAllowedModal(true);
        } else {
          setFeedback(error || "Failed to start OAuth flow");
        }
        resetCaptcha();
        setIsGoogleLoading(false);
        return;
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      resetCaptcha();
      setIsGoogleLoading(false);
      setFeedback(
        error instanceof Error ? error.message : "Failed to start OAuth flow",
      );
    }
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);
    if (isCloudEnv && !turnstile.verified && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "info",
      });

      setIsLoading(false);
      resetCaptcha();
      return;
    }

    if (data.email.includes("@agpt.co")) {
      toast({
        title: "Please use Google SSO to login using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      resetCaptcha();
      return;
    }

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: data.email,
          password: data.password,
          turnstileToken: turnstile.token,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        toast({
          title: result?.error || "Login failed",
          variant: "destructive",
        });
        setIsLoading(false);
        resetCaptcha();
        turnstile.reset();
        return;
      }

      await supabase?.auth.refreshSession();
      setIsLoading(false);
      setFeedback(null);

      const next =
        (result?.next as string) || (result?.onboarding ? "/onboarding" : "/");
      if (next) router.push(next);
    } catch (error) {
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during login",
        variant: "destructive",
      });
      setIsLoading(false);
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
    handleSubmit: form.handleSubmit(handleLogin),
    handleProviderLogin,
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
  };
}
