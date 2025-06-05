import { useTurnstile } from "@/hooks/useTurnstile";
import useSupabase from "@/lib/supabase/useSupabase";
import { signupFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { signup } from "./actions";
import { providerLogin } from "../login/actions";
import z from "zod";

export function useSignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const isProdEnv = process.env.NODE_ENV === "production";

  const turnstile = useTurnstile({
    action: "signup",
    autoVerify: false,
    resetOnError: true,
  });

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
    const error = await providerLogin(provider);
    setIsGoogleLoading(false);
    if (error) {
      setFeedback(error);
      return;
    }
    setFeedback(null);
  }

  async function handleSignup(data: z.infer<typeof signupFormSchema>) {
    setIsLoading(true);

    if (!turnstile.verified) {
      setFeedback("Please complete the CAPTCHA challenge.");
      setIsLoading(false);
      return;
    }

    if (data.email.includes("@agpt.co")) {
      setFeedback(
        "Please use Google SSO to create an account using an AutoGPT email.",
      );

      setIsLoading(false);
      return;
    }

    const error = await signup(data, turnstile.token as string);
    setIsLoading(false);
    if (error) {
      if (error === "user_already_exists") {
        setFeedback("User with this email already exists");
        turnstile.reset();
        return;
      } else {
        setFeedback(error);
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
    isLoggedIn: !!user,
    isLoading,
    isProdEnv,
    isUserLoading,
    isGoogleLoading,
    isSupabaseAvailable: !!supabase,
    handleSubmit: form.handleSubmit(handleSignup),
    handleProviderSignup,
  };
}
