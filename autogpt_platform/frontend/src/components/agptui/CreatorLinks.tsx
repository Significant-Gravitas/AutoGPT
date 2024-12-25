import * as React from "react";
import { getIconForSocial } from "@/components/ui/icons";

interface CreatorLinksProps {
  links: string[];
}

export const CreatorLinks: React.FC<CreatorLinksProps> = ({ links }) => {
  if (!links || links.length === 0) {
    return null;
  }

  const renderLinkButton = (url: string) => (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex min-w-[200px] flex-1 items-center justify-between rounded-[34px] border border-neutral-600 px-5 py-3 dark:border-neutral-400"
    >
      <div className="font-neue text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        {new URL(url).hostname.replace("www.", "")}
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
      <div className="font-neue text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        Other links
      </div>
      <div className="flex w-full flex-wrap gap-3">
        {links.map((link, index) => (
          <React.Fragment key={index}>{renderLinkButton(link)}</React.Fragment>
        ))}
      </div>
    </div>
  );
};
