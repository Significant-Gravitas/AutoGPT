import { AITeamIcon } from "@/components/atoms/AITeamIcon/AITeamIcon";
import { Button } from "@/components/atoms/Button/Button";

/** What a signed-in user sees before AI Experts reach their account. */
export function ExpertComingSoon() {
  return (
    <section className="flex flex-col items-center py-20 text-center sm:py-28">
      <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)] ring-1 ring-zinc-200">
        <AITeamIcon size={26} className="text-zinc-900" />
      </span>
      <h1 className="mt-6 text-2xl font-semibold tracking-[-0.02em] text-zinc-900">
        Coming soon
      </h1>
      <p className="mt-2 max-w-sm text-[15px] leading-6 text-zinc-500">
        Ready-made specialists you hire in a click, working for you in minutes.
        This expert&apos;s page opens as soon as Experts reach your account.
      </p>
      <div className="mt-8">
        <Button
          as="NextLink"
          href="/marketplace"
          variant="secondary"
          size="small"
        >
          Back to marketplace
        </Button>
      </div>
    </section>
  );
}
