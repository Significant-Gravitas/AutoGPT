"use client";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { environment } from "@/services/environment";
import { LoadingLogin } from "./components/LoadingLogin";
import { useLoginPage } from "./useLoginPage";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { useSearchParams } from "next/navigation";

export default function LoginPage() {
  const searchParams = useSearchParams();
  const nextUrl = searchParams.get("next");
  // Preserve next parameter when switching between login/signup
  const signupHref = nextUrl
    ? `/signup?next=${encodeURIComponent(nextUrl)}`
    : "/signup";

  const {
    user,
    form,
    feedback,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isSupabaseAvailable,
    handleSubmit,
    handleProviderLogin,
    handleCloseNotAllowedModal,
  } = useLoginPage();

  if (isUserLoading || user) {
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
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      <AuthCard title="Login to your account">
        <Form {...form}>
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
            <FormField
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
            <FormField
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
        </Form>
        <AuthCard.BottomText
          text="Don't have an account?"
          link={{ text: "Sign up", href: signupHref }}
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
