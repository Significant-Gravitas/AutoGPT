import * as React from "react";
import { IconExternalLink } from "@/components/ui/icons";

interface CreatorLinksProps {
  links: {
    website?: string;
    linkedin?: string;
    github?: string;
    other?: string[];
  };
}

export const CreatorLinks: React.FC<CreatorLinksProps> = ({ links }) => {
  const hasLinks = links.website || links.linkedin || links.github || (links.other && links.other.length > 0);

  if (!hasLinks) {
    return null;
  }

  const renderLinkButton = (text: string, href?: string) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex min-w-[200px] flex-1 items-center justify-between rounded-[34px] border border-neutral-600 px-5 py-3"
    >
      <div className="font-neue text-base font-medium leading-normal text-neutral-800">
        {text}
      </div>
      <div className="relative h-6 w-6">
        <IconExternalLink className="h-6 w-6 text-neutral-800" />
      </div>
    </a>
  );

  return (
    <div className="flex flex-col items-start justify-start gap-4">
      <div className="font-neue text-base font-medium leading-normal text-neutral-800">
        Other links
      </div>
      <div className="flex w-full flex-wrap gap-3">
        {links.website && renderLinkButton("Website link", links.website)}
        {links.linkedin && renderLinkButton("LinkedIn", links.linkedin)}
        {links.github && renderLinkButton("GitHub", links.github)}
        {links.other?.map((link, index) => (
          <React.Fragment key={index}>
            {renderLinkButton("Other links", link)}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};