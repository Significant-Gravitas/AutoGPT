import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { providerLogin } from "../login/actions";
import { signup } from "./actions";
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

    if (!turnstile.verified && !isVercelPreview) {
      toast({
        title: "Please complete the CAPTCHA challenge.",
        variant: "default",
      });
      setIsGoogleLoading(false);
      resetCaptcha();
      return;
    }

    const error = await providerLogin(provider);
    if (error) {
      setIsGoogleLoading(false);
      resetCaptcha();
      toast({
        title: error,
        variant: "destructive",
      });
      return;
    }
    setFeedback(null);
  }

  async function handleSignup(data: z.infer<typeof signupFormSchema>) {
    setIsLoading(true);

    if (!turnstile.verified && !isVercelPreview) {
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

    const error = await signup(data, turnstile.token as string);
    setIsLoading(false);
    if (error) {
      if (error === "user_already_exists") {
        setFeedback("User with this email already exists");
        turnstile.reset();
        return;
      } else if (error === "not_allowed") {
        setShowNotAllowedModal(true);
      } else {
        toast({
          title: error,
          variant: "destructive",
        });
        resetCaptcha();
        turnstile.reset();
      }
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
    handleSubmit: form.handleSubmit(handleSignup),
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
    handleProviderSignup,
  };
}
