import { useTurnstile } from "@/hooks/useTurnstile";
import useSupabase from "@/lib/supabase/useSupabase";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { login, providerLogin } from "./actions";
import z from "zod";

export function useLoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);

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

  useEffect(() => {
    if (user) router.push("/");
  }, [user]);

  async function handleProviderLogin(provider: LoginProvider) {
    setIsGoogleLoading(true);
    const error = await providerLogin(provider);
    setIsGoogleLoading(false);
    if (error) {
      setFeedback(error);
      return;
    }
    setFeedback(null);
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);

    if (!(await form.trigger())) {
      setIsLoading(false);
      return;
    }

    if (!turnstile.verified) {
      setFeedback("Please complete the CAPTCHA challenge.");
      setIsLoading(false);
      return;
    }

    const error = await login(data, turnstile.token as string);
    await supabase?.auth.refreshSession();
    setIsLoading(false);
    if (error) {
      setFeedback(error);
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
    isLoggedIn: !!user,
    isLoading,
    isUserLoading,
    isGoogleLoading,
    isSupabaseAvailable: !!supabase,
    handleLogin,
    handleProviderLogin,
  };
}
