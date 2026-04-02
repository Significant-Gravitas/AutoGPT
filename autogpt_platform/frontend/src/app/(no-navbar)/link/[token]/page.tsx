"use client";

import { Button } from "@/components/atoms/Button/Button";
import { AuthCard } from "@/components/auth/AuthCard";
import { Text } from "@/components/atoms/Text/Text";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { CheckCircle, LinkBreak, Spinner } from "@phosphor-icons/react";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

/** Platform display names */
const PLATFORM_NAMES: Record<string, string> = {
  DISCORD: "Discord",
  TELEGRAM: "Telegram",
  SLACK: "Slack",
  TEAMS: "Teams",
  WHATSAPP: "WhatsApp",
  GITHUB: "GitHub",
  LINEAR: "Linear",
};

type LinkState =
  | { status: "loading" }
  | { status: "not-authenticated" }
  | { status: "ready" }
  | { status: "linking" }
  | { status: "success"; platform: string }
  | { status: "error"; message: string };

export default function PlatformLinkPage() {
  const params = useParams();
  const token = params.token as string;
  const { user, supabase, isUserLoading } = useSupabase();

  const [state, setState] = useState<LinkState>({ status: "loading" });

  // Determine initial state based on auth.
  // Token validity is checked server-side on confirm — no need for a
  // separate status call (which requires bot API key anyway).
  useEffect(() => {
    if (!token) return;
    // Wait for Supabase to finish loading before deciding auth state
    if (isUserLoading) return;

    if (!user) {
      setState({ status: "not-authenticated" });
    } else {
      setState({ status: "ready" });
    }
  }, [token, user, isUserLoading]);

  async function handleLink() {
    setState({ status: "linking" });

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
      // The proxy injects auth from the server-side Supabase session cookie,
      // so we don't need to send Authorization ourselves.
      const res = await fetch(
        `/api/proxy/api/platform-linking/tokens/${token}/confirm`,
        {
          method: "POST",
          body: JSON.stringify({}),
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
        },
      );

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        const message =
          data?.detail ??
          "Failed to link your account. The link may have expired.";
        setState({ status: "error", message });
        return;
      }

      const data = await res.json();
      const platformName = PLATFORM_NAMES[data.platform] ?? data.platform;
      setState({ status: "success", platform: platformName });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setState({
          status: "error",
          message:
            "Request timed out. Please go back to your chat and try again.",
        });
      } else {
        setState({
          status: "error",
          message: "Something went wrong. Please try again.",
        });
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      {state.status === "loading" && <LoadingView />}

      {state.status === "not-authenticated" && (
        <NotAuthenticatedView token={token} />
      )}

      {state.status === "ready" && <ReadyView onLink={handleLink} />}

      {state.status === "linking" && <LinkingView />}

      {state.status === "success" && <SuccessView platform={state.platform} />}

      {state.status === "error" && <ErrorView message={state.message} />}

      <div className="mt-8 text-center text-xs text-muted-foreground">
        <p>Powered by AutoGPT Platform</p>
      </div>
    </div>
  );
}

function LoadingView() {
  return (
    <AuthCard title="Link your account">
      <div className="flex flex-col items-center gap-4">
        <Spinner size={48} className="animate-spin text-primary" />
        <Text variant="body-medium" className="text-muted-foreground">
          Verifying link...
        </Text>
      </div>
    </AuthCard>
  );
}

function NotAuthenticatedView({ token }: { token: string }) {
  const loginUrl = `/login?next=${encodeURIComponent(`/link/${token}`)}`;

  return (
    <AuthCard title="Link your account">
      <div className="flex w-full flex-col items-center gap-6">
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          Sign in to your AutoGPT account to link it with your chat platform.
        </Text>
        <Button as="NextLink" href={loginUrl} className="w-full">
          Sign in to continue
        </Button>
        <AuthCard.BottomText
          text="Don't have an account?"
          link={{ text: "Sign up", href: `/signup?next=/link/${token}` }}
        />
      </div>
    </AuthCard>
  );
}

function ReadyView({ onLink }: { onLink: () => void }) {
  return (
    <AuthCard title="Link your account">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="rounded-xl bg-slate-50 p-6 text-center">
          <Text variant="body-medium" className="text-muted-foreground">
            Connect your chat platform account to AutoGPT to use CoPilot.
          </Text>
        </div>
        <Button onClick={onLink} className="w-full">
          Link account
        </Button>
      </div>
    </AuthCard>
  );
}

function LinkingView() {
  return (
    <AuthCard title="Linking...">
      <div className="flex flex-col items-center gap-4">
        <Spinner size={48} className="animate-spin text-primary" />
        <Text variant="body-medium" className="text-muted-foreground">
          Connecting your accounts...
        </Text>
      </div>
    </AuthCard>
  );
}

function SuccessView({ platform }: { platform: string }) {
  return (
    <AuthCard title="Account linked!">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <CheckCircle size={40} weight="fill" className="text-green-600" />
        </div>
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          Your <strong>{platform}</strong> account is now linked to AutoGPT.
          <br />
          You can close this page and go back to your chat.
        </Text>
      </div>
    </AuthCard>
  );
}

function ErrorView({ message }: { message: string }) {
  return (
    <AuthCard title="Link failed">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
          <LinkBreak size={40} weight="bold" className="text-red-600" />
        </div>
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          {message}
        </Text>
        <Text variant="small" className="text-center text-muted-foreground">
          Go back to your chat and ask the bot for a new link.
        </Text>
      </div>
    </AuthCard>
  );
}
