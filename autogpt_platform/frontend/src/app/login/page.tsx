"use client";
import { login, providerLogin } from "./actions";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import useSupabase from "@/hooks/useSupabase";
import Spinner from "@/components/Spinner";
import {
  AuthCard,
  AuthHeader,
  AuthButton,
  AuthFeedback,
  AuthBottomText,
  PasswordInput,
} from "@/components/auth";
import { loginFormSchema } from "@/types/auth";

export default function LoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  // TODO: uncomment when we enable social login
  // const onProviderLogin = useCallback(async (
  //   provider: LoginProvider,
  // ) => {
  //   setIsLoading(true);
  //   const error = await providerLogin(provider);
  //   setIsLoading(false);
  //   if (error) {
  //     setFeedback(error);
  //     return;
  //   }
  //   setFeedback(null);
  // }, [supabase]);

  const onLogin = useCallback(
    async (data: z.infer<typeof loginFormSchema>) => {
      setIsLoading(true);

      if (!(await form.trigger())) {
        setIsLoading(false);
        return;
      }

      const error = await login(data);
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        return;
      }
      setFeedback(null);
    },
    [form],
  );

  if (user) {
    console.debug("User exists, redirecting to /");
    router.push("/");
  }

  if (isUserLoading || user) {
    return <Spinner />;
  }

  if (!supabase) {
    return (
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  return (
    <AuthCard className="mx-auto">
      <AuthHeader>Login to your account</AuthHeader>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onLogin)}>
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem className="mb-6">
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input
                    placeholder="m@example.com"
                    {...field}
                    type="email" // Explicitly specify email type
                    autoComplete="username" // Added for password managers
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem className="mb-6">
                <FormLabel className="flex w-full items-center justify-between">
                  <span>Password</span>
                  <Link
                    href="/reset_password"
                    className="text-sm font-normal leading-normal text-black underline"
                  >
                    Forgot your password?
                  </Link>
                </FormLabel>
                <FormControl>
                  <PasswordInput
                    {...field}
                    autoComplete="current-password" // Added for password managers
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <AuthButton
            onClick={() => onLogin(form.getValues())}
            isLoading={isLoading}
            type="submit"
          >
            Login
          </AuthButton>
        </form>
        <AuthFeedback message={feedback} isError={true} />
      </Form>
      <AuthBottomText
        text="Don't have an account?"
        linkText="Sign up"
        href="/signup"
      />
    </AuthCard>
  );
}
