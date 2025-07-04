"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleLoadingModal } from "@/components/auth/GoogleLoadingModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import Turnstile from "@/components/auth/Turnstile";
import { Form, FormField } from "@/components/ui/form";
import { getBehaveAs } from "@/lib/utils";
import { LoadingLogin } from "./components/LoadingLogin";
import { useLoginPage } from "./useLoginPage";

export default function LoginPage() {
  const {
    form,
    feedback,
    turnstile,
    captchaKey,
    isLoading,
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
    return <LoadingLogin />;
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
        {true ? (
          <>
            <div className="mb-3 w-full">
              <GoogleOAuthButton
                onClick={() => handleProviderLogin("google")}
                isLoading={isGoogleLoading}
                disabled={isLoading}
              />
            </div>
            <div className="mb-3 flex w-full items-center">
              <div className="h-px flex-1 bg-gray-300"></div>
              <span className="text-md mx-3 text-gray-500">or</span>
              <div className="h-px flex-1 bg-gray-300"></div>
            </div>
          </>
        ) : null}
        <Form {...form}>
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <Input
                  label="Email"
                  placeholder="m@example.com"
                  type="email"
                  autoComplete="username"
                  className="w-full"
                  error={form.formState.errors.email?.message}
                  {...field}
                />
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <Input
                  label="Password"
                  placeholder="•••••••••••••••••••••"
                  type="password"
                  autoComplete="current-password"
                  error={form.formState.errors.password?.message}
                  hint={
                    <Link variant="secondary" href="/reset-password">
                      Forgot password?
                    </Link>
                  }
                  {...field}
                />
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

            <Button
              variant="primary"
              loading={isLoading}
              type="submit"
              className="mt-6 w-full"
            >
              Login
            </Button>
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
