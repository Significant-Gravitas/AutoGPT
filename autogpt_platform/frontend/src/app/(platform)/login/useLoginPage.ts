import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { login, providerLogin } from "./actions";
import z from "zod";
import { BehaveAs } from "@/lib/utils";
import { getBehaveAs } from "@/lib/utils";

export function useLoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const [captchaKey, setCaptchaKey] = useState(0);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const isCloudEnv = getBehaveAs() === BehaveAs.CLOUD;

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
    try {
      const error = await providerLogin(provider);
      if (error) throw error;
      setFeedback(null);
    } catch (error) {
      resetCaptcha();
      setFeedback(JSON.stringify(error));
    } finally {
      setIsGoogleLoading(false);
    }
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);
    if (!turnstile.verified) {
      setFeedback("Please complete the CAPTCHA challenge.");
      setIsLoading(false);
      resetCaptcha();
      return;
    }

    if (data.email.includes("@agpt.co")) {
      setFeedback("Please use Google SSO to login using an AutoGPT email.");
      setIsLoading(false);
      resetCaptcha();
      return;
    }

    const error = await login(data, turnstile.token as string);
    await supabase?.auth.refreshSession();
    setIsLoading(false);
    if (error) {
      setFeedback(error);
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
    isSupabaseAvailable: !!supabase,
    handleSubmit: form.handleSubmit(handleLogin),
    handleProviderLogin,
  };
}
