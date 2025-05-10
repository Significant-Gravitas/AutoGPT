import * as React from "react";
import CreatorLink from "./CreatorLink";

interface CreatorLinksProps {
  links: string[];
}

export const CreatorLinks: React.FC<CreatorLinksProps> = ({ links }) => {
  if (!links || links.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="font-sans text-base font-medium text-zinc-800">
        Other links
      </div>
      <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2">
        {links.map((link, index) => (
          <CreatorLink href={link} key={index}>
            {new URL(link).hostname.replace("www.", "").replace(".com", "")}
          </CreatorLink>
        ))}
      </div>
    </div>
  );
};
