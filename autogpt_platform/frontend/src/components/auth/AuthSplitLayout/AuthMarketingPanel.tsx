import { ReactNode } from "react";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Text } from "@/components/atoms/Text/Text";

interface FeatureItem {
  icon: ReactNode;
  title: string;
  description?: string;
}

interface Props {
  heading: ReactNode;
  description?: string;
  itemsTitle?: string;
  items: FeatureItem[];
  footerText?: string;
}

export function AuthMarketingPanel({
  heading,
  description,
  itemsTitle,
  items,
  footerText = "Trusted by builders and teams",
}: Props) {
  return (
    <div className="relative flex h-full w-full flex-col justify-between p-12 xl:p-16">
      <div className="absolute inset-0 -z-0 bg-[radial-gradient(circle_at_top_left,rgba(99,102,241,0.18),transparent_55%),radial-gradient(circle_at_bottom_right,rgba(168,85,247,0.18),transparent_60%)]" />
      <div className="relative z-10 flex flex-col gap-12">
        <AutoGPTLogoMark />
        <div className="flex flex-col gap-3">
          <Text
            variant="h1"
            as="h1"
            className="!leading-[1.1] tracking-[-0.02em] !text-white"
          >
            {heading}
          </Text>
          {description ? (
            <Text variant="large" className="max-w-sm !text-slate-300">
              {description}
            </Text>
          ) : null}
        </div>
        {itemsTitle ? (
          <Text
            variant="small-medium"
            className="uppercase tracking-[0.14em] !text-slate-400"
          >
            {itemsTitle}
          </Text>
        ) : null}
        <ul className="flex flex-col gap-5">
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
      <div className="relative z-10 mt-12 flex items-center gap-3">
        <TrustAvatars />
        <Text variant="small-medium" className="!text-slate-400">
          {footerText}
        </Text>
      </div>
    </div>
  );
}

function AutoGPTLogoMark() {
  return (
    <div className="flex items-center gap-2">
      <AutoGPTLogo hideText className="h-8 w-auto" />
      <span className="font-poppins text-xl font-semibold tracking-tight text-white">
        AutoGPT
      </span>
    </div>
  );
}

const trustAvatarColors = [
  "bg-rose-400",
  "bg-amber-400",
  "bg-emerald-400",
  "bg-sky-400",
  "bg-violet-400",
];

function TrustAvatars() {
  return (
    <div className="flex -space-x-2">
      {trustAvatarColors.map((color, idx) => (
        <span
          key={color}
          className={`inline-block h-7 w-7 rounded-full border-2 border-slate-950 ${color}`}
          style={{ zIndex: trustAvatarColors.length - idx }}
        />
      ))}
    </div>
  );
}
