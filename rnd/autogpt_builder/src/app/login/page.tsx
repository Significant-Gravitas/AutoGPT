"use client";
import useUser from "@/hooks/useUser";
import { login, signup } from "./actions";
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
import { useSupabase } from "@/components/SupabaseProvider";
import { useRouter } from "next/navigation";

const loginFormSchema = z.object({
  email: z.string().email().min(2).max(64),
  password: z.string().min(6).max(64),
});

export default function LoginPage() {
  const { supabase, isLoading: isSupabaseLoading } = useSupabase();
  const { user, isLoading: isUserLoading } = useUser();
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

  if (user) {
    console.log("User exists, redirecting to profile");
    router.push("/profile");
  }

  if (isUserLoading || isSupabaseLoading || user) {
    return (
      <div className="flex justify-center items-center h-[80vh]">
        <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
      </div>
    );
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
        // Get Google provider_refresh_token
        // queryParams: {
        //   access_type: 'offline',
        //   prompt: 'consent',
        // },
      },
    });

    if (!error) {
      setFeedback(null);
      return;
    }
    setFeedback(error.message);
  }

  const onLogin = async (data: z.infer<typeof loginFormSchema>) => {
    setIsLoading(true);
    const error = await login(data);
    setIsLoading(false);
    if (error) {
      setFeedback(error);
      return;
    }
    setFeedback(null);
  };

  const onSignup = async (data: z.infer<typeof loginFormSchema>) => {
    if (await form.trigger()) {
      setIsLoading(true);
      const error = await signup(data);
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        return;
      }
      setFeedback(null);
    }
  };

  return (
    <div className="flex items-center justify-center h-[80vh]">
      <div className="w-full max-w-md p-8 rounded-lg shadow-md space-y-6">
        <div className="mb-6 space-y-2">
          <Button
            className="w-full"
            onClick={() => handleSignInWithProvider("google")}
            variant="outline"
            type="button"
            disabled={isLoading}
          >
            <FaGoogle className="mr-2 h-4 w-4" />
            Sign in with Google
          </Button>
          <Button
            className="w-full"
            onClick={() => handleSignInWithProvider("github")}
            variant="outline"
            type="button"
            disabled={isLoading}
          >
            <FaGithub className="mr-2 h-4 w-4" />
            Sign in with GitHub
          </Button>
          <Button
            className="w-full"
            onClick={() => handleSignInWithProvider("discord")}
            variant="outline"
            type="button"
            disabled={isLoading}
          >
            <FaDiscord className="mr-2 h-4 w-4" />
            Sign in with Discord
          </Button>
        </div>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onLogin)}>
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <FormItem className="mb-4">
                  <FormLabel>Email</FormLabel>
                  <FormControl>
                    <Input placeholder="user@email.com" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Password</FormLabel>
                  <FormControl>
                    <PasswordInput placeholder="password" {...field} />
                  </FormControl>
                  <FormDescription>
                    Password needs to be at least 6 characters long
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="flex w-full space-x-4 mt-6 mb-6">
              <Button
                className="w-1/2 flex justify-center"
                type="submit"
                disabled={isLoading}
              >
                Log in
              </Button>
              <Button
                className="w-1/2 flex justify-center"
                variant="outline"
                type="button"
                onClick={form.handleSubmit(onSignup)}
                disabled={isLoading}
              >
                Sign up
              </Button>
            </div>
          </form>
          <p className="text-red-500 text-sm">{feedback}</p>
          <p className="text-primary text-center text-sm">
            By continuing you agree to everything
          </p>
        </Form>
      </div>
    </div>
  );
}
