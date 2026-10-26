import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";

const INTEGRATIONS = [
  { id: "github", name: "GitHub" },
  { id: "google", name: "Google" },
  { id: "notion", name: "Notion" },
  { id: "slack", name: "Slack" },
  { id: "hubspot", name: "HubSpot" },
  { id: "linear", name: "Linear" },
  { id: "airtable", name: "Airtable" },
  { id: "discord", name: "Discord" },
];

interface Props {
  footerText: string;
}

export function MarketingFooter({ footerText }: Props) {
  return (
    <div className="flex flex-col gap-3">
      <Text
        variant="small-medium"
        // TODO(tokens): no exact text token for tracking-[0.14em]
        className="uppercase tracking-[0.14em] !text-foreground-tertiary"
      >
        {footerText}
      </Text>
      <ul className="flex flex-wrap gap-2">
        {INTEGRATIONS.map(({ id, name }) => (
          <li
            key={id}
            className="flex size-11 items-center justify-center rounded-xl bg-surface smooth-shadow-ring-sm"
          >
            <Image
              src={`/integrations/${id}.png`}
              alt={name}
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
