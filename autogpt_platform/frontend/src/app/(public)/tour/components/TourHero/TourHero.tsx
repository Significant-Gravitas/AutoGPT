import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRightIcon, SparkleIcon } from "@phosphor-icons/react/dist/ssr";

export function TourHero() {
  return (
    <section className="flex flex-1 items-center justify-center px-6 py-16">
      <div className="flex w-full max-w-2xl flex-col items-center text-center">
        <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-4 py-1.5">
          <SparkleIcon className="h-4 w-4 text-violet-600" weight="fill" />
          <Text variant="small-medium" className="text-zinc-600">
            Live demo — no signup required
          </Text>
        </span>

        <Text as="h1" variant="h1" className="text-balance">
          Build AI agents by just chatting
        </Text>

        <Text
          variant="lead"
          className="mt-5 max-w-xl text-balance text-zinc-600"
        >
          Describe a goal in plain English and watch Autopilot build and run a
          working AutoGPT agent for you — in seconds, right in your browser.
        </Text>

        <div className="mt-10 flex w-full flex-col items-center justify-center gap-3 sm:w-auto sm:flex-row">
          <Button
            as="NextLink"
            href="/tour/chat"
            size="large"
            rightIcon={<ArrowRightIcon className="h-4 w-4" weight="bold" />}
            className="w-full sm:w-auto"
          >
            Try the demo
          </Button>
          <Button
            as="NextLink"
            href="https://agpt.co/pricing"
            target="_blank"
            rel="noopener noreferrer"
            variant="ghost"
            size="large"
            className="w-full sm:w-auto"
          >
            See pricing
          </Button>
        </div>
      </div>
    </section>
  );
}
