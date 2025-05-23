import providers from "../../../common/providers.json";
import {
  FaGithub,
  FaGoogle,
  FaDiscord,
  FaMedium,
  FaHubspot,
  FaTwitter,
  FaKey,
} from "react-icons/fa";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import type { FC } from "react";

export type ProviderMeta = {
  id: string;
  display_name: string;
  icon?: string;
};
export const providersList = providers as ProviderMeta[];

export const PROVIDER_NAMES = providersList.reduce(
  (acc, p) => {
    (acc as any)[p.id.toUpperCase()] = p.id;
    return acc;
  },
  {} as Record<string, string>,
) as {
  [K in keyof any]: string;
};
export type CredentialsProviderName = (typeof providersList)[number]["id"];

const fallbackIcon = FaKey;
const iconMap: Record<string, FC<{ className?: string }>> = {
  FaGithub,
  FaGoogle,
  FaDiscord,
  FaMedium,
  FaHubspot,
  FaTwitter,
  NotionLogoIcon,
  fallbackIcon,
};

export const providerIcons = providersList.reduce(
  (acc, p) => {
    acc[p.id as CredentialsProviderName] = p.icon
      ? iconMap[p.icon] || fallbackIcon
      : fallbackIcon;
    return acc;
  },
  {} as Record<CredentialsProviderName, FC<{ className?: string }>>,
);

export const providerDisplayNames = providersList.reduce(
  (acc, p) => {
    acc[p.id as CredentialsProviderName] = p.display_name;
    return acc;
  },
  {} as Record<CredentialsProviderName, string>,
);
