const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
// Paths that carry the ID as the next segment: /embed/ID, /shorts/ID, ...
const ID_PATH_PREFIXES = new Set(["embed", "v", "e", "shorts", "live"]);

function parseHttpUrl(url: string): URL | null {
  const trimmed = url.trim();
  const withScheme = /^[a-z][a-z\d+.-]*:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
  try {
    const parsed = new URL(withScheme);
    return parsed.protocol === "https:" || parsed.protocol === "http:"
      ? parsed
      : null;
  } catch {
    return null;
  }
}

// Pages that play a video. Image hosts (img.youtube.com, i.ytimg.com) are
// left alone, so a thumbnail stays an image.
const YOUTUBE_PAGE_HOSTS = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "music.youtube.com",
  "youtube-nocookie.com",
  "www.youtube-nocookie.com",
  "youtu.be",
  "www.youtu.be",
]);

function isYouTubeHost(hostname: string): boolean {
  return YOUTUBE_PAGE_HOSTS.has(hostname.toLowerCase());
}

function parseYouTubeUrl(url: string): URL | null {
  const parsed = parseHttpUrl(url);
  return parsed && isYouTubeHost(parsed.hostname) ? parsed : null;
}

/** Any link to YouTube, whether or not it names a single video. */
export function isYouTubeUrl(url: string): boolean {
  return parseYouTubeUrl(url) !== null;
}

export function getYouTubeVideoId(url: string): string | null {
  const parsed = parseYouTubeUrl(url);
  if (!parsed) return null;
  const segments = parsed.pathname.split("/").filter(Boolean);
  const host = parsed.hostname.toLowerCase();
  let candidate: string | null | undefined;
  if (host === "youtu.be" || host === "www.youtu.be") {
    candidate = segments[0];
  } else if (segments[0] === "watch") {
    candidate = parsed.searchParams.get("v");
  } else if (segments[0] && ID_PATH_PREFIXES.has(segments[0])) {
    candidate = segments[1];
  }
  return candidate && VIDEO_ID.test(candidate) ? candidate : null;
}

// Privacy-enhanced mode: YouTube sets no cookies until the viewer presses play.
export function getYouTubeEmbedUrl(videoId: string): string {
  return `https://www.youtube-nocookie.com/embed/${encodeURIComponent(videoId)}`;
}

/** Where "Watch on YouTube" goes: the video's page, or the link as given. */
export function getYouTubeWatchUrl(url: string): string | null {
  const videoId = getYouTubeVideoId(url);
  if (videoId) {
    return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;
  }
  return parseYouTubeUrl(url)?.toString() ?? null;
}

export function getNoticeText({
  hasVideoId,
  canAllowHere,
}: {
  hasVideoId: boolean;
  canAllowHere: boolean;
}): string {
  if (!hasVideoId) return "This video can only be watched on YouTube.";
  if (canAllowHere) {
    return "YouTube sets advertising cookies when this video loads. To watch it here, allow marketing cookies.";
  }
  return "YouTube sets advertising cookies when this video loads.";
}
