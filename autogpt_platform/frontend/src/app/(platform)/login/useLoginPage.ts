import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { login, providerLogin } from "./actions";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { TurnstileInstance } from "@marsidev/react-turnstile";

export function useLoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaRef, setCaptchaRef] = useState<TurnstileInstance | null>(null);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const isCloudEnv = getBehaveAs() === BehaveAs.CLOUD;
  const isVercelPreview = process.env.NEXT_PUBLIC_VERCEL_ENV === "preview";

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  useEffect(() => {
    if (user) router.push("/");
  }, [user]);

  async function handleProviderLogin(provider: LoginProvider) {
    setIsGoogleLoading(true);

    if (isCloudEnv && !captchaToken && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "info",
      });

      setIsGoogleLoading(false);
      return;
    }

    try {
      const error = await providerLogin(provider);
      if (error) throw error;
      setFeedback(null);
    } catch (error) {
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
    if (isCloudEnv && !captchaToken && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "info",
      });

      setIsLoading(false);
      return;
    }

    if (data.email.includes("@agpt.co")) {
      toast({
        title: "Please use Google SSO to login using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      return;
    }

    const error = await login(data, captchaToken as string);
    await supabase?.auth.refreshSession();
    setIsLoading(false);
    if (error) {
      toast({
        title: error,
        variant: "destructive",
      });

      return;
    }
    setFeedback(null);
  }

  function handleCaptchaVerify(token: string) {
    setCaptchaToken(token);
  }

  function handleCaptchaReady(ref: TurnstileInstance) {
    setCaptchaRef(ref);
  }

  return {
    form,
    feedback,
    captchaRef,
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
    handleCaptchaVerify,
    handleCaptchaReady,
  };
}
