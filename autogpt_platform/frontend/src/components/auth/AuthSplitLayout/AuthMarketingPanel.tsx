import { ReactNode } from "react";
import { AutoGPTLogoWhite } from "@/components/atoms/AutoGPTLogo/AutoGPTLogoWhite";
import { Text } from "@/components/atoms/Text/Text";
import { IntegrationsMarquee } from "@/components/molecules/IntegrationsMarquee/IntegrationsMarquee";
import { Vortex } from "@/components/ui/vortex";
import { AnimatedHeading } from "./AnimatedHeading";

interface FeatureItem {
  icon: ReactNode;
  title: string;
  description?: string;
}

interface Props {
  headingLines: ReactNode[];
  description?: string;
  itemsTitle?: string;
  items: FeatureItem[];
  footerText?: string;
}

export function AuthMarketingPanel({
  headingLines,
  description,
  itemsTitle,
  items,
  footerText = "Connects with the tools you already use",
}: Props) {
  return (
    <Vortex
      containerClassName="absolute inset-0 h-full w-full"
      className="relative flex h-full w-full flex-col justify-between px-12 pb-12 pt-10 xl:px-16 xl:pb-16"
      backgroundColor="transparent"
      particleCount={350}
      baseHue={260}
      rangeSpeed={1.2}
      baseRadius={1}
      rangeRadius={2}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_30%,rgba(2,6,23,0.55)_85%)]"
      />
      <div className="relative z-10 flex flex-col gap-12">
        <div className="relative isolate w-fit">
          <div
            aria-hidden
            className="absolute inset-0 -z-10 scale-[1.8] rounded-full bg-[radial-gradient(circle,rgba(139,92,246,0.55),rgba(99,102,241,0.25)_45%,transparent_70%)] blur-2xl"
          />
          <AutoGPTLogoWhite
            hideText
            className="h-auto w-[3.5rem] drop-shadow-[0_0_18px_rgba(167,139,250,0.45)]"
          />
        </div>
        <div className="flex flex-col gap-4">
          <Text
            variant="h1"
            as="h1"
            className="!w-full !text-[3rem] !font-semibold !leading-[1.05] tracking-[-0.025em] !text-white"
          >
            <AnimatedHeading lines={headingLines} />
          </Text>
          {description ? (
            <Text variant="large" className="max-w-md !text-slate-300">
              {description}
            </Text>
          ) : null}
        </div>
        <div className="flex flex-col gap-2">
          {itemsTitle ? (
            <Text
              variant="small-medium"
              className="uppercase tracking-[0.14em] !text-slate-400"
            >
              {itemsTitle}
            </Text>
          ) : null}
          <ul className="flex flex-col gap-3">
            {items.map((item) => (
              <li
                key={item.title}
                className="flex items-start gap-4 rounded-xl border border-white/5 bg-white/[0.03] p-4 backdrop-blur-sm"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/30 to-purple-500/30 text-white ring-1 ring-white/10">
                  {item.icon}
                </span>
                <div className="flex flex-col gap-0.5">
                  <Text variant="body-medium" className="!text-white">
                    {item.title}
                  </Text>
                  {item.description ? (
                    <Text variant="small" className="!text-slate-400">
                      {item.description}
                    </Text>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="relative z-10 mt-12 flex flex-col gap-3">
        <Text
          variant="small-medium"
          className="uppercase tracking-[0.14em] !text-slate-400"
        >
          {footerText}
        </Text>
        <IntegrationsMarquee
          variant="dark"
          className="h-[140px] w-full max-w-[460px]"
        />
      </div>
    </Vortex>
  );
}
