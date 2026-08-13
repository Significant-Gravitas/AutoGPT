import { Text } from "@/components/atoms/Text/Text";
import { IntegrationsMarquee } from "@/components/molecules/IntegrationsMarquee/IntegrationsMarquee";

interface Props {
  footerText: string;
}

export function MarketingFooter({ footerText }: Props) {
  return (
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
  );
}
