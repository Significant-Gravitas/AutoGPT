import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CookieSettingsLink } from "@/components/molecules/CookieSettingsLink/CookieSettingsLink";
import { TourBackground } from "./TourBackground";
import { ArrowRight02Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function TourHero() {
  return (
    <section className="relative flex flex-1 flex-col overflow-hidden border-b border-separator">
      <TourBackground />
      <div className="relative z-10 flex flex-1 flex-col items-center overflow-y-auto px-6 pb-4 pt-16">
        <div className="flex w-full max-w-2xl flex-1 flex-col items-center justify-center text-center">
          <AutoGPTLogo
            hideText
            viewBox="46 0 43 40"
            className="mb-6 h-20 w-20"
          />

          {/* oxlint-disable-next-line design-system/no-arbitrary-values -- hero glass badge's indigo gradient ring and glow, no token */}
          <span className="relative mb-6 inline-flex overflow-hidden rounded-full bg-[linear-gradient(135deg,rgba(99,102,241,0.6),rgba(59,130,246,0.35),rgba(165,180,252,0.6))] p-px shadow-[0_6px_20px_-6px_rgba(59,130,246,0.35)]">
            {/* oxlint-disable-next-line design-system/no-arbitrary-values -- glass highlight, no shadow token */}
            <span className="relative inline-flex items-center gap-2 rounded-full bg-surface/50 px-4 py-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)] backdrop-blur-xl">
              <span className="contents text-brand">
                <Icon icon={SparklesIcon} className="h-4 w-4" />
              </span>
              <Text variant="body-medium-tight" tone="body">
                Live demo, no signup required
              </Text>
            </span>
          </span>

          <Text
            as="h1"
            variant="h1"
            // oxlint-disable-next-line design-system/no-restyle, design-system/no-arbitrary-values -- marketing hero size, awaiting product decision (SECRT-3152)
            className="text-balance text-[4rem] leading-none"
          >
            Build AI agents by just chatting
          </Text>

          <Text
            variant="lead"
            tone="secondary"
            className="mt-5 max-w-xl text-balance"
          >
            Describe a goal in plain English and watch Otto build and run a
            working AutoGPT workflow for you, in seconds, right in your browser.
          </Text>

          <div className="mt-10 flex w-full flex-col items-center justify-center gap-3 sm:w-auto sm:flex-row">
            <Button
              as="NextLink"
              href="/tour/chat"
              size="large"
              rightIcon={
                <span className="flex transition-transform duration-200 ease-out group-hover:translate-x-1">
                  <Icon icon={ArrowRight02Icon} className="h-4 w-4" />
                </span>
              }
              // oxlint-disable-next-line design-system/no-restyle -- the hero CTA is wider than the large size; no Button size for it
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
        <CookieSettingsLink className="mt-6" />
      </div>
    </section>
  );
}
