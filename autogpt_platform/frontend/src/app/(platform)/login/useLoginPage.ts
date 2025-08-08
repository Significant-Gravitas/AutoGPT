import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { login, providerLogin } from "./actions";
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
      const error = await providerLogin(provider);
      if (error) throw error;
      setFeedback(null);
    } catch (error) {
      resetCaptcha();
      setIsGoogleLoading(false);
      const errorString = JSON.stringify(error);
      if (errorString.includes("not_allowed")) {
        setShowNotAllowedModal(true);
      } else {
        setFeedback(errorString);
      }
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

    const error = await login(data, turnstile.token as string);
    await supabase?.auth.refreshSession();
    setIsLoading(false);
    if (error) {
      toast({
        title: error,
        variant: "destructive",
      });

      resetCaptcha();
      // Always reset the turnstile on any error
      turnstile.reset();
      return;
    }
    setFeedback(null);
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
