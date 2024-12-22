"use client";
import { signup, signupFormSchema } from "./actions";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { PasswordInput } from "@/components/PasswordInput";
import { FaGoogle, FaGithub, FaDiscord, FaSpinner } from "react-icons/fa";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Checkbox } from "@/components/ui/checkbox";
import useSupabase from "@/hooks/useSupabase";
import Spinner from "@/components/Spinner";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import AuthCard from "@/components/auth/AuthCard";
import AuthHeader from "@/components/auth/AuthHeader";
import AuthButton from "@/components/auth/AuthButton";
import AuthBottomText from "@/components/auth/AuthBottomText";
import AuthFeedback from "@/components/auth/AuthFeedback";

export default function LoginPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const api = useBackendAPI();

  const form = useForm<z.infer<typeof signupFormSchema>>({
    resolver: zodResolver(signupFormSchema),
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
      agreeToTerms: false,
    },
  });

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

  async function handleSignInWithProvider(
    provider: "google" | "github" | "discord",
  ) {
    const { data, error } = await supabase!.auth.signInWithOAuth({
      provider: provider,
      options: {
        redirectTo:
          process.env.AUTH_CALLBACK_URL ??
          `http://localhost:3000/auth/callback`,
      },
    });

    await api.createUser();

    if (!error) {
      setFeedback(null);
      return;
    }
    setFeedback(error.message);
  }

  const onSignup = async (data: z.infer<typeof signupFormSchema>) => {
    setIsLoading(true);
    const error = await signup(data);
    setIsLoading(false);
    if (error) {
      setFeedback(error);
      return;
    }
    setFeedback(null);
  };

  return (
    <AuthCard>
      <AuthHeader>Create a new account</AuthHeader>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSignup)}>
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem className="mb-6">
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input placeholder="m@example.com" {...field} />
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
                <FormLabel>Password</FormLabel>
                <FormControl>
                  <PasswordInput {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="confirmPassword"
            render={({ field }) => (
              <FormItem className="mb-4">
                <FormLabel>Confirm Password</FormLabel>
                <FormControl>
                  <PasswordInput {...field} />
                </FormControl>
                <FormDescription className="text-slate-500 text-sm font-normal leading-tight">
                  Password needs to be at least 6 characters long
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          <AuthButton
            onClick={() => onSignup(form.getValues())}
            isLoading={isLoading}
            type="submit"
          >
            Sign up
          </AuthButton>
          <FormField
            control={form.control}
            name="agreeToTerms"
            render={({ field }) => (
              <FormItem className="mt-6 flex flex-row items-start space-x-2 -space-y-1">
                <FormControl>
                  <Checkbox
                    checked={field.value}
                    onCheckedChange={field.onChange}
                  />
                </FormControl>
                <div className="">
                  <FormLabel>
                    <span className="mr-1 text-slate-950 text-sm font-normal leading-normal">I agree to the</span>
                    <Link
                      href="https://auto-gpt.notion.site/Terms-of-Use-11400ef5bece80d0b087d7831c5fd6bf"
                      className="text-slate-950 text-sm font-normal leading-normal underline"
                    >
                      Terms of Use
                    </Link>
                    <span className="mx-1 text-slate-950 text-sm font-normal leading-normal">and</span>
                    <Link
                      href="https://www.notion.so/auto-gpt/Privacy-Policy-ab11c9c20dbd4de1a15dcffe84d77984"
                      className="text-slate-950 text-sm font-normal leading-normal underline"
                    >
                      Privacy Policy
                    </Link>
                  </FormLabel>
                  <FormMessage />
                </div>
              </FormItem>
            )}
          />
        </form>
        <AuthFeedback message={feedback} isError={true}/>
      </Form>
      <AuthBottomText
        text="Already a member?"
        linkText="Log in"
        href="/login"
      />
    </AuthCard>
  );
}
