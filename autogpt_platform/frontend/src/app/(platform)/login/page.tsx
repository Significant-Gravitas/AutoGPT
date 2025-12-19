"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { environment } from "@/services/environment";
import { Controller, FormProvider } from "react-hook-form";
import { LoadingLogin } from "./components/LoadingLogin";
import { useLoginPage } from "./useLoginPage";

export default function LoginPage() {
  const {
    user,
    form,
    feedback,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isAuthAvailable,
    handleSubmit,
    handleProviderLogin,
    handleCloseNotAllowedModal,
  } = useLoginPage();

  if (isUserLoading || user) {
    return <LoadingLogin />;
  }

  if (!isAuthAvailable) {
    return <div>User accounts are disabled because auth is unavailable</div>;
  }

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      <AuthCard title="Login to your account">
        <FormProvider {...form}>
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
            <Controller
              control={form.control}
              name="email"
              render={({ field }) => (
                <Input
                  id={field.name}
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
            <Controller
              control={form.control}
              name="password"
              render={({ field }) => (
                <Input
                  id={field.name}
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

            <Button
              variant="primary"
              loading={isLoading}
              disabled={isGoogleLoading}
              type="submit"
              className="mt-6 w-full"
            >
              {isLoading ? "Logging in..." : "Login"}
            </Button>
          </form>
          {isCloudEnv ? (
            <GoogleOAuthButton
              onClick={() => handleProviderLogin("google")}
              isLoading={isGoogleLoading}
              disabled={isLoading}
            />
          ) : null}
          <AuthFeedback
            type="login"
            message={feedback}
            isError={!!feedback}
            behaveAs={environment.getBehaveAs()}
          />
        </FormProvider>
        <AuthCard.BottomText
          text="Don't have an account?"
          link={{ text: "Sign up", href: "/signup" }}
        />
      </AuthCard>
      <MobileWarningBanner />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </div>
  );
}
