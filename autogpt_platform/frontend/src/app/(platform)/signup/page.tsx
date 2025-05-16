"use client";
import { signup } from "./actions";
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
import type { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Checkbox } from "@/components/ui/checkbox";
import useSupabase from "@/hooks/useSupabase";
import LoadingBox from "@/components/ui/loading";
import {
  AuthCard,
  AuthHeader,
  AuthButton,
  AuthBottomText,
  PasswordInput,
  Turnstile,
} from "@/components/auth";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { signupFormSchema } from "@/types/auth";
import { getBehaveAs } from "@/lib/utils";
import { useTurnstile } from "@/hooks/useTurnstile";

export default function SignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  //TODO: Remove after closed beta

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

  const onSignup = useCallback(
    async (data: z.infer<typeof signupFormSchema>) => {
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
    },
    [form, turnstile],
  );

  if (user) {
    console.debug("User exists, redirecting to /");
    router.push("/");
  }

  if (isUserLoading || user) {
    return <LoadingBox className="h-[80vh]" />;
  }

  if (!supabase) {
    return (
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  return (
    <AuthCard className="mx-auto mt-12">
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
                  <Input
                    placeholder="m@example.com"
                    {...field}
                    type="email"
                    autoComplete="email"
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
                <FormLabel>Password</FormLabel>
                <FormControl>
                  <PasswordInput {...field} autoComplete="new-password" />
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
                  <PasswordInput {...field} autoComplete="new-password" />
                </FormControl>
                <FormDescription className="text-sm font-normal leading-tight text-slate-500">
                  Password needs to be at least 6 characters long
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Turnstile CAPTCHA Component */}
          <Turnstile
            siteKey={turnstile.siteKey}
            onVerify={turnstile.handleVerify}
            onExpire={turnstile.handleExpire}
            onError={turnstile.handleError}
            action="signup"
            shouldRender={turnstile.shouldRender}
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
              <FormItem className="mt-6 flex flex-row items-start -space-y-1 space-x-2">
                <FormControl>
                  <Checkbox
                    checked={field.value}
                    onCheckedChange={field.onChange}
                  />
                </FormControl>
                <div className="">
                  <FormLabel>
                    <span className="mr-1 text-sm font-normal leading-normal text-slate-950">
                      I agree to the
                    </span>
                    <Link
                      href="https://auto-gpt.notion.site/Terms-of-Use-11400ef5bece80d0b087d7831c5fd6bf"
                      className="text-sm font-normal leading-normal text-slate-950 underline"
                    >
                      Terms of Use
                    </Link>
                    <span className="mx-1 text-sm font-normal leading-normal text-slate-950">
                      and
                    </span>
                    <Link
                      href="https://www.notion.so/auto-gpt/Privacy-Policy-ab11c9c20dbd4de1a15dcffe84d77984"
                      className="text-sm font-normal leading-normal text-slate-950 underline"
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
      </Form>
      <AuthFeedback
        type="signup"
        message={feedback}
        isError={!!feedback}
        behaveAs={getBehaveAs()}
      />

      <AuthBottomText
        text="Already a member?"
        linkText="Log in"
        href="/login"
      />
    </AuthCard>
  );
}
