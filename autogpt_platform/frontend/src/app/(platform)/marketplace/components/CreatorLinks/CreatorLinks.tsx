import { getIconForSocial } from "@/components/__legacy__/ui/icons";
import { Fragment } from "react";

interface CreatorLinksProps {
  links: string[];
}

function normalizeURL(url: string): string {
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    return `https://${url}`;
  }
  return url;
}

function getHostnameFromURL(url: string): string {
  try {
    const normalizedURL = normalizeURL(url);
    return new URL(normalizedURL).hostname.replace("www.", "");
  } catch {
    return url.replace(/^(https?:\/\/)?(www\.)?/, "");
  }
}

export const CreatorLinks = ({ links }: CreatorLinksProps) => {
  if (!links || links.length === 0) {
    return null;
  }

  const renderLinkButton = (url: string) => (
    <a
      href={normalizeURL(url)}
      target="_blank"
      rel="noopener noreferrer"
      className="flex min-w-[200px] flex-1 items-center justify-between rounded-[34px] border border-neutral-600 px-5 py-3 dark:border-neutral-400"
    >
      <div className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        {getHostnameFromURL(url)}
      </div>
      <div className="relative h-6 w-6">
        {getIconForSocial(url, {
          className: "h-6 w-6 text-neutral-800 dark:text-neutral-200",
        })}
      </div>
    </a>
  );

  return (
    <div className="flex flex-col items-start justify-start gap-4">
      <div className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        Other links
      </div>
      <div className="flex w-full flex-wrap gap-3">
        {links.map((link, index) => (
          <Fragment key={index}>{renderLinkButton(link)}</Fragment>
        ))}
      </div>
    </div>
  );
};
