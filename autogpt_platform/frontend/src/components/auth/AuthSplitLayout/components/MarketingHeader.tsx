import { ReactNode } from "react";
import { AutoGPTLogoWhite } from "@/components/atoms/AutoGPTLogo/AutoGPTLogoWhite";
import { Text } from "@/components/atoms/Text/Text";
import { AnimatedHeading } from "../AnimatedHeading";

interface Props {
  headingLines: ReactNode[];
  description?: string;
}

export function MarketingHeader({ headingLines, description }: Props) {
  return (
    <>
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
    </>
  );
}
