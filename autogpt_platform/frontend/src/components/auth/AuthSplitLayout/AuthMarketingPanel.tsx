import { ReactNode } from "react";
import { Vortex } from "@/components/ui/vortex";
import { FeatureList } from "./components/FeatureList";
import { FeatureItem } from "./components/FeatureListItem";
import { MarketingFooter } from "./components/MarketingFooter";
import { MarketingHeader } from "./components/MarketingHeader";

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
      baseSpeed={0.08}
      rangeSpeed={0.22}
      baseRadius={1}
      rangeRadius={2}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_30%,rgba(2,6,23,0.55)_85%)]"
      />
      <div className="relative z-10 flex flex-col gap-12">
        <MarketingHeader
          headingLines={headingLines}
          description={description}
        />
        <FeatureList itemsTitle={itemsTitle} items={items} />
      </div>
      <MarketingFooter footerText={footerText} />
    </Vortex>
  );
}
