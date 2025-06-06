"use client";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { Checkbox } from "@/components/ui/checkbox";
import LoadingBox from "@/components/ui/loading";
import {
  AuthCard,
  AuthHeader,
  AuthButton,
  AuthBottomText,
  GoogleOAuthButton,
  PasswordInput,
  Turnstile,
} from "@/components/auth";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { getBehaveAs } from "@/lib/utils";
import { useSignupPage } from "./useSignupPage";

export default function SignupPage() {
  const {
    form,
    feedback,
    turnstile,
    captchaKey,
    isLoggedIn,
    isLoading,
    isCloudEnv,
    isUserLoading,
    isGoogleLoading,
    isSupabaseAvailable,
    handleSubmit,
    handleProviderSignup,
  } = useSignupPage();

  if (isUserLoading || isLoggedIn) {
    return <LoadingBox className="h-[80vh]" />;
  }

  if (!isSupabaseAvailable) {
    return (
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  return (
    <AuthCard className="mx-auto mt-12">
      <AuthHeader>Create a new account</AuthHeader>

      {isCloudEnv ? (
        <>
          <div className="mb-6">
            <GoogleOAuthButton
              onClick={() => handleProviderSignup("google")}
              isLoading={isGoogleLoading}
              disabled={isLoading}
            />
          </div>
          <div className="mb-6 flex items-center">
            <div className="flex-1 border-t border-gray-300"></div>
            <span className="mx-3 text-sm text-gray-500">or</span>
            <div className="flex-1 border-t border-gray-300"></div>
          </div>
        </>
      ) : null}

      <Form {...form}>
        <form onSubmit={handleSubmit}>
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
                  Password needs to be at least 12 characters long
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Turnstile CAPTCHA Component */}
          <Turnstile
            key={captchaKey}
            siteKey={turnstile.siteKey}
            onVerify={turnstile.handleVerify}
            onExpire={turnstile.handleExpire}
            onError={turnstile.handleError}
            setWidgetId={turnstile.setWidgetId}
            action="signup"
            shouldRender={turnstile.shouldRender}
          />

          <AuthButton isLoading={isLoading} type="submit">
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
