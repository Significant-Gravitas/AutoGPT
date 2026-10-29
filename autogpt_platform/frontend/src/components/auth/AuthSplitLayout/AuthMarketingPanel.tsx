import { ReactNode } from "react";
import { FeatureList } from "./components/FeatureList";
import { FeatureItem } from "./components/FeatureListItem";
import { MarketingFooter } from "./components/MarketingFooter";
import { MarketingHeader } from "./components/MarketingHeader";

interface Props {
  headingLines: ReactNode[];
  description?: string;
  itemsTitle?: string;
  items?: FeatureItem[];
  footerText?: string;
  /** Shown under the header in place of the feature list. */
  children?: ReactNode;
}

export function AuthMarketingPanel({
  headingLines,
  description,
  itemsTitle,
  items = [],
  footerText = "Connects with the tools you already use",
  children,
}: Props) {
  return (
    <div className="flex h-full w-full flex-col justify-between gap-12 px-12 pb-12 pt-10 xl:px-16 xl:pb-16">
      <div className="flex flex-col gap-12">
        <MarketingHeader
          headingLines={headingLines}
          description={description}
        />
        {children ?? <FeatureList itemsTitle={itemsTitle} items={items} />}
      </div>
      <MarketingFooter footerText={footerText} />
    </div>
  );
}
