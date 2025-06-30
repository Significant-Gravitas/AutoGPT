"use client";
import AuthButton from "@/components/auth/AuthButton";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleLoadingModal } from "@/components/auth/GoogleLoadingModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { PasswordInput } from "@/components/auth/PasswordInput";
import Turnstile from "@/components/auth/Turnstile";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import LoadingBox from "@/components/ui/loading";
import { getBehaveAs } from "@/lib/utils";
import Link from "next/link";
import { useLoginPage } from "./useLoginPage";

export default function LoginPage() {
  const {
    form,
    feedback,
    turnstile,
    captchaKey,
    isLoading,
    isCloudEnv,
    isLoggedIn,
    isUserLoading,
    isGoogleLoading,
    showNotAllowedModal,
    isSupabaseAvailable,
    handleSubmit,
    handleProviderLogin,
    handleCloseNotAllowedModal,
  } = useLoginPage();

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
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center">
      <AuthCard title="Login to your account">
        {isCloudEnv ? (
          <>
            <div className="mb-6">
              <GoogleOAuthButton
                onClick={() => handleProviderLogin("google")}
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
          <form onSubmit={handleSubmit} className="w-full">
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
                      href="/reset-password"
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

            {/* Turnstile CAPTCHA Component */}
            <Turnstile
              key={captchaKey}
              siteKey={turnstile.siteKey}
              onVerify={turnstile.handleVerify}
              onExpire={turnstile.handleExpire}
              onError={turnstile.handleError}
              setWidgetId={turnstile.setWidgetId}
              action="login"
              shouldRender={turnstile.shouldRender}
            />

            <AuthButton isLoading={isLoading} type="submit">
              Login
            </AuthButton>
          </form>
          <AuthFeedback
            type="login"
            message={feedback}
            isError={!!feedback}
            behaveAs={getBehaveAs()}
          />
        </Form>
        <AuthCard.BottomText
          text="Don't have an account?"
          link={{ text: "Sign up", href: "/signup" }}
        />
      </AuthCard>
      <GoogleLoadingModal isOpen={isGoogleLoading} />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </div>
  );
}
