import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRightIcon, SparkleIcon } from "@phosphor-icons/react/dist/ssr";
import { TourBackground } from "./TourBackground";

export function TourHero() {
  return (
    <section className="relative flex flex-1 flex-col items-center justify-center overflow-hidden border-b border-neutral-100 px-6 py-16">
      <TourBackground />
      <div className="relative z-10 flex w-full max-w-2xl flex-col items-center text-center">
        <AutoGPTLogo hideText viewBox="46 0 43 40" className="mb-6 h-20 w-20" />

        <span className="relative mb-6 inline-flex overflow-hidden rounded-full bg-[linear-gradient(135deg,rgba(99,102,241,0.6),rgba(59,130,246,0.35),rgba(165,180,252,0.6))] p-px shadow-[0_6px_20px_-6px_rgba(59,130,246,0.35)]">
          <span className="relative inline-flex items-center gap-2 rounded-full bg-white/50 px-4 py-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)] backdrop-blur-xl">
            <SparkleIcon className="h-4 w-4 text-violet-600" weight="fill" />
            <Text variant="small-medium" className="text-sm text-zinc-700">
              Live demo, no signup required
            </Text>
          </span>
        </span>

        <Text
          as="h1"
          variant="h1"
          className="text-balance text-[4rem] font-[600] leading-[1]"
        >
          Build AI agents by just chatting
        </Text>

        <Text
          variant="lead"
          className="mt-5 max-w-xl text-balance text-zinc-600"
        >
          Describe a goal in plain English and watch Autopilot build and run a
          working AutoGPT agent for you, in seconds, right in your browser.
        </Text>

        <div className="mt-10 flex w-full flex-col items-center justify-center gap-3 sm:w-auto sm:flex-row">
          <Button
            as="NextLink"
            href="/tour/chat"
            size="large"
            rightIcon={
              <ArrowRightIcon
                className="h-4 w-4 transition-transform duration-200 ease-out group-hover:translate-x-1"
                weight="bold"
              />
            }
            className="group w-full sm:w-auto sm:px-8"
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
