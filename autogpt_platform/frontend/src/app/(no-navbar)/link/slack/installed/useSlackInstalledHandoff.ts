import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";

// Long enough for the browser to hand the deep link to the Slack app, short
// enough that nobody sits looking at a handoff screen.
const RETURN_DELAY_MS = 1200;

export function useSlackInstalledHandoff() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const team = searchParams.get("team") ?? "";
  const app = searchParams.get("app") ?? "";
  const bot = searchParams.get("bot") ?? "";

  // The desktop scheme opens the bot DM without navigating this tab, so we
  // stay in control and can send the user back to their settings. The https
  // form is the fallback for anyone without the Slack app.
  const deepLink = bot
    ? `slack://user?team=${encodeURIComponent(team)}&id=${encodeURIComponent(bot)}`
    : "";
  const openSlackHref = `https://slack.com/app_redirect?app=${encodeURIComponent(app)}&team=${encodeURIComponent(team)}`;

  function openSlack() {
    if (deepLink) window.location.href = deepLink;
    else window.open(openSlackHref, "_blank", "noopener,noreferrer");
  }

  useEffect(() => {
    if (deepLink) window.location.href = deepLink;
    const timer = window.setTimeout(
      () => router.replace("/settings/bots"),
      RETURN_DELAY_MS,
    );
    return () => window.clearTimeout(timer);
  }, [deepLink, router]);

  return { openSlackHref, openSlack };
}
