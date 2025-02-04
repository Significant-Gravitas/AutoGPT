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
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Checkbox } from "@/components/ui/checkbox";
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
import { signupFormSchema } from "@/types/auth";

export default function SignupPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  //TODO: Remove after closed beta
  const [showWaitlistPrompt, setShowWaitlistPrompt] = useState(false);

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

      const error = await signup(data);
      setIsLoading(false);
      if (error) {
        if (error === "user_already_exists") {
          setFeedback("User with this email already exists");
          return;
        } else {
          setShowWaitlistPrompt(true);
        }
        return;
      }
      setFeedback(null);
      setShowWaitlistPrompt(false);
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
        <AuthFeedback message={feedback} isError={true} />
      </Form>
      {showWaitlistPrompt && (
        <div>
          <span className="mr-1 text-sm font-normal leading-normal text-red-500">
            The provided email may not be allowed to sign up.
          </span>
          <br />
          <span className="mx-1 text-sm font-normal leading-normal text-slate-950">
            - AutoGPT Platform is currently in closed beta. You can join
          </span>
          <Link
            href="https://agpt.co/waitlist"
            className="text-sm font-normal leading-normal text-slate-950 underline"
          >
            the waitlist here.
          </Link>
          <br />
          <span className="mx-1 text-sm font-normal leading-normal text-slate-950">
            - Make sure you use the same email address you used to sign up for
            the waitlist.
          </span>
          <br />
          <span className="mx-1 text-sm font-normal leading-normal text-slate-950">
            - You can self host the platform, visit our
          </span>
          <Link
            href="https://agpt.co/waitlist"
            className="text-sm font-normal leading-normal text-slate-950 underline"
          >
            GitHub repository.
          </Link>
        </div>
      )}
      <AuthBottomText
        text="Already a member?"
        linkText="Log in"
        href="/login"
      />
    </AuthCard>
  );
}
