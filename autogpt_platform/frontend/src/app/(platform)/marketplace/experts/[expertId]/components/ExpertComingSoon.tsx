import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  isLoggedIn: boolean;
}

/** What a visitor sees before AI Experts are open to them: signed out, or
 *  signed in without the feature yet. */
export function ExpertComingSoon({ isLoggedIn }: Props) {
  return (
    <section className="relative flex flex-col items-center gap-6 overflow-hidden rounded-3xl border border-zinc-200/60 bg-[linear-gradient(180deg,rgba(139,92,246,0.10),rgba(139,92,246,0.03)_60%,transparent)] px-6 py-16 text-center sm:py-24">
      <span className="inline-flex items-center gap-2 rounded-full bg-white/80 px-3 py-1 text-xs font-medium uppercase tracking-[0.14em] text-violet-700 ring-1 ring-inset ring-violet-600/10">
        Coming soon
      </span>
      <AITeamIcon size={56} />
      <div className="flex max-w-lg flex-col gap-2">
        <Text variant="h3" className="text-zinc-900">
          Meet the AI Experts
        </Text>
        <Text variant="large" className="text-zinc-600">
          Ready-made specialists you can hire in a click, competent on day one
          and working for you in minutes. This expert&apos;s page opens as soon
          as Experts reach your account.
        </Text>
      </div>
      <div className="flex flex-col items-center gap-3 sm:flex-row">
        {isLoggedIn ? null : (
          <Button as="NextLink" href="/login" variant="primary" size="large">
            Sign in
          </Button>
        )}
        <Button
          as="NextLink"
          href="/marketplace"
          variant="secondary"
          size="large"
        >
          Back to marketplace
        </Button>
      </div>
    </section>
  );
}
