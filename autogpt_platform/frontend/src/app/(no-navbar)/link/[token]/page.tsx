"use client";

import { Button } from "@/components/atoms/Button/Button";
import { AuthCard } from "@/components/auth/AuthCard";
import { Text } from "@/components/atoms/Text/Text";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

/** Platform display names and icons */
const PLATFORM_INFO: Record<string, { name: string; icon: string }> = {
  DISCORD: { name: "Discord", icon: "🎮" },
  TELEGRAM: { name: "Telegram", icon: "✈️" },
  SLACK: { name: "Slack", icon: "💬" },
  TEAMS: { name: "Teams", icon: "👥" },
  WHATSAPP: { name: "WhatsApp", icon: "📱" },
  GITHUB: { name: "GitHub", icon: "🐙" },
  LINEAR: { name: "Linear", icon: "📐" },
};

type LinkState =
  | { status: "loading" }
  | { status: "not-authenticated" }
  | { status: "ready"; platform: string; platformUsername?: string }
  | { status: "linking" }
  | { status: "success"; platform: string }
  | { status: "error"; message: string };

export default function PlatformLinkPage() {
  const params = useParams();
  const token = params.token as string;
  const { user, supabase } = useSupabaseStore();

  const [state, setState] = useState<LinkState>({ status: "loading" });

  // Check token validity on mount
  useEffect(() => {
    if (!token) return;

    async function checkToken() {
      try {
        const res = await fetch(`/api/proxy/api/platform-linking/tokens/${token}/status`);
        if (!res.ok) {
          setState({
            status: "error",
            message: "This link is invalid or has expired.",
          });
          return;
        }

        const data = await res.json();

        if (data.status === "linked") {
          setState({
            status: "success",
            platform: "your platform",
          });
          return;
        }

        if (data.status === "expired") {
          setState({
            status: "error",
            message: "This link has expired. Please ask the bot for a new one.",
          });
          return;
        }

        // Token is pending — check if user is logged in
        if (!user) {
          setState({ status: "not-authenticated" });
          return;
        }

        // Fetch token details to show platform info
        // The status endpoint doesn't return platform info,
        // so we show a generic prompt
        setState({
          status: "ready",
          platform: "your platform",
        });
      } catch {
        setState({
          status: "error",
          message: "Something went wrong. Please try again.",
        });
      }
    }

    checkToken();
  }, [token, user]);

  // Handle the link confirmation
  const handleLink = useCallback(async () => {
    if (!supabase) return;

    setState({ status: "linking" });

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session?.access_token) {
        setState({ status: "not-authenticated" });
        return;
      }

      const res = await fetch(`/api/proxy/api/platform-linking/tokens/${token}/confirm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${session.access_token}`,
        },
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        const message =
          data?.detail ??
          "Failed to link your account. The link may have expired.";
        setState({ status: "error", message });
        return;
      }

      const data = await res.json();
      const platformInfo = PLATFORM_INFO[data.platform];
      setState({
        status: "success",
        platform: platformInfo?.name ?? data.platform,
      });
    } catch {
      setState({
        status: "error",
        message: "Something went wrong. Please try again.",
      });
    }
  }, [token, supabase]);

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      {state.status === "loading" && <LoadingView />}

      {state.status === "not-authenticated" && (
        <NotAuthenticatedView token={token} />
      )}

      {state.status === "ready" && (
        <ReadyView
          platform={state.platform}
          platformUsername={state.platformUsername}
          onLink={handleLink}
        />
      )}

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
        <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-primary" />
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

function ReadyView({
  platform,
  platformUsername,
  onLink,
}: {
  platform: string;
  platformUsername?: string;
  onLink: () => void;
}) {
  return (
    <AuthCard title="Link your account">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="rounded-xl bg-slate-50 p-6 text-center">
          <Text variant="body-medium" className="text-muted-foreground">
            Connect your <strong>{platform}</strong> account
            {platformUsername && (
              <>
                {" "}
                (<strong>{platformUsername}</strong>)
              </>
            )}{" "}
            to your AutoGPT account to use CoPilot.
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
        <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-primary" />
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
          <span className="text-3xl">✅</span>
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
          <span className="text-3xl">❌</span>
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
