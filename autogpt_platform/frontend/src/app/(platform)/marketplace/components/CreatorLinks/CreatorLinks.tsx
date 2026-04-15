import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  FacebookLogo,
  GithubLogo,
  Globe,
  InstagramLogo,
  LinkedinLogo,
  TiktokLogo,
  XLogo,
  YoutubeLogo,
} from "@phosphor-icons/react";

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

function getSocialIcon(url: string) {
  let host;
  try {
    host = new URL(normalizeURL(url)).hostname.toLowerCase();
  } catch {
    return <Globe className="h-4 w-4" />;
  }

  if (host === "facebook.com" || host.endsWith(".facebook.com")) {
    return <FacebookLogo className="h-4 w-4" />;
  } else if (
    host === "twitter.com" ||
    host.endsWith(".twitter.com") ||
    host === "x.com" ||
    host.endsWith(".x.com")
  ) {
    return <XLogo className="h-4 w-4" />;
  } else if (host === "instagram.com" || host.endsWith(".instagram.com")) {
    return <InstagramLogo className="h-4 w-4" />;
  } else if (host === "linkedin.com" || host.endsWith(".linkedin.com")) {
    return <LinkedinLogo className="h-4 w-4" />;
  } else if (host === "github.com" || host.endsWith(".github.com")) {
    return <GithubLogo className="h-4 w-4" />;
  } else if (host === "youtube.com" || host.endsWith(".youtube.com")) {
    return <YoutubeLogo className="h-4 w-4" />;
  } else if (host === "tiktok.com" || host.endsWith(".tiktok.com")) {
    return <TiktokLogo className="h-4 w-4" />;
  }
  return <Globe className="h-4 w-4" />;
}

export function CreatorLinks({ links }: CreatorLinksProps) {
  if (!links || links.length === 0) return null;

  return (
    <div className="flex flex-col items-start gap-3">
      <Text variant="h5">Links</Text>
      <div className="flex flex-wrap gap-2">
        {links.map((link) => (
          <Button
            key={link}
            variant="secondary"
            size="small"
            as="NextLink"
            href={normalizeURL(link)}
            target="_blank"
            rel="noopener noreferrer"
            rightIcon={getSocialIcon(link)}
          >
            {getHostnameFromURL(link)}
          </Button>
        ))}
      </div>
    </div>
  );
}
