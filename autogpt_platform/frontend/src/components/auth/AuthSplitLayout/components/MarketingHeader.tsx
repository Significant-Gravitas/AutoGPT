import { ReactNode } from "react";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Text } from "@/components/atoms/Text/Text";
import { TeamAvatars } from "./TeamAvatars";

interface Props {
  headingLines: ReactNode[];
  description?: string;
}

export function MarketingHeader({ headingLines, description }: Props) {
  return (
    <>
      <AutoGPTLogo className="h-auto w-28" />
      <div className="flex flex-col gap-6">
        <TeamAvatars />
        <div className="flex flex-col gap-4">
          <Text
            variant="h1"
            as="h1"
            tone="primary"
            className="!w-full !text-display !leading-tight !font-semibold tracking-heading"
          >
            {headingLines.map((line, i) => (
              <span key={i} className="block">
                {line}
              </span>
            ))}
          </Text>
          {description ? (
            <Text variant="large" tone="muted" className="max-w-md">
              {description}
            </Text>
          ) : null}
        </div>
      </div>
    </>
  );
}
