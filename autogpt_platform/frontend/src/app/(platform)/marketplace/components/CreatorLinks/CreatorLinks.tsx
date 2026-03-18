import { getIconForSocial } from "@/components/__legacy__/ui/icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

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

export function CreatorLinks({ links }: CreatorLinksProps) {
  if (!links || links.length === 0) return null;

  return (
    <div className="flex flex-col items-start gap-3">
      <Text variant="h5">Links</Text>
      <div className="flex flex-wrap gap-2">
        {links.map((link, index) => (
          <Button
            key={index}
            variant="secondary"
            size="small"
            as="NextLink"
            href={normalizeURL(link)}
            target="_blank"
            rel="noopener noreferrer"
            rightIcon={getIconForSocial(link, {
              className: "h-4 w-4",
            })}
          >
            {getHostnameFromURL(link)}
          </Button>
        ))}
      </div>
    </div>
  );
}
