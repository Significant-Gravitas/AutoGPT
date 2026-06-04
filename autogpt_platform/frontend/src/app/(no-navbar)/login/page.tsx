"use client";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { AuthDivider } from "@/components/auth/AuthSplitLayout/AuthDivider";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout/AuthSplitLayout";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { environment } from "@/services/environment";
import { useSearchParams } from "next/navigation";
import { LoadingLogin } from "./components/LoadingLogin";
import { LoginMarketingPanel } from "./components/LoginMarketingPanel";
import { useLoginPage } from "./useLoginPage";

export const dynamic = "force-dynamic";

export default function LoginPage() {
  const searchParams = useSearchParams();
  const nextUrl = searchParams.get("next");
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
      <AuthSplitLayout marketing={<LoginMarketingPanel />}>
        <Text variant="body-medium" className="text-center !text-slate-500">
          User accounts are disabled because Supabase client is unavailable
        </Text>
      </AuthSplitLayout>
    );
  }

  return (
    <AuthSplitLayout marketing={<LoginMarketingPanel />}>
      <div className="mb-8">
        <Text variant="h3" as="h1" className="!text-slate-950">
          Log in to your account to continue
        </Text>
      </div>

      <Form {...form}>
        <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Email"
                placeholder="name@company.com"
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
                placeholder="Enter your password"
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
            {isLoading ? "Logging in..." : "Log in"}
          </Button>
        </form>

        {isCloudEnv ? (
          <>
            <AuthDivider />
            <GoogleOAuthButton
              onClick={() => handleProviderLogin("google")}
              isLoading={isGoogleLoading}
              disabled={isLoading}
            />
          </>
        ) : null}

        <AuthFeedback
          type="login"
          message={feedback}
          isError={!!feedback}
          behaveAs={environment.getBehaveAs()}
        />
      </Form>

      <div className="mt-6 inline-flex w-full items-center justify-center gap-1">
        <Text variant="body-medium" className="!text-slate-500">
          Don&apos;t have an account?
        </Text>
        <Link href={signupHref} variant="secondary">
          Sign up
        </Link>
      </div>

      <MobileWarningBanner />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </AuthSplitLayout>
  );
}
