import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";

const INTEGRATIONS = [
  "github",
  "google",
  "notion",
  "slack",
  "hubspot",
  "linear",
  "airtable",
  "discord",
];

interface Props {
  footerText: string;
}

export function MarketingFooter({ footerText }: Props) {
  return (
    <div className="flex flex-col gap-3">
      <Text
        variant="small-medium"
        className="uppercase tracking-[0.14em] !text-zinc-400"
      >
        {footerText}
      </Text>
      <ul aria-hidden className="flex flex-wrap gap-2">
        {INTEGRATIONS.map((id) => (
          <li
            key={id}
            className="flex size-11 items-center justify-center rounded-xl bg-white smooth-shadow-ring-sm"
          >
            <Image
              src={`/integrations/${id}.png`}
              alt=""
              width={22}
              height={22}
              className="size-[22px] rounded-sm object-contain"
            />
          </li>
        ))}
      </ul>
    </div>
  );
}
