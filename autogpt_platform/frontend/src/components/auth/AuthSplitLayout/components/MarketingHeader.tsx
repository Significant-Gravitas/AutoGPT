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
            // TODO(tokens): SECRT-3152, rare size !text-[3rem]; no exact text token for !leading-[1.05]
            className="!w-full !text-[3rem] !font-semibold !leading-[1.05] tracking-heading !text-foreground-strong"
          >
            {headingLines.map((line, i) => (
              <span key={i} className="block">
                {line}
              </span>
            ))}
          </Text>
          {description ? (
            <Text
              variant="large"
              className="max-w-md !text-foreground-tertiary"
            >
              {description}
            </Text>
          ) : null}
        </div>
      </div>
    </>
  );
}
