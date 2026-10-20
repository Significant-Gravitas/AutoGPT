const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
// Paths that carry the ID as the next segment: /embed/ID, /shorts/ID, ...
const ID_PATH_PREFIXES = new Set(["embed", "v", "e", "shorts", "live"]);
// Embed paths for a playlist or a channel's stream, not one video, though
// they are 11 characters long like an ID.
const NOT_VIDEO_IDS = new Set(["videoseries", "live_stream"]);

function parseHttpUrl(url: string): URL | null {
  const trimmed = url.trim();
  const withScheme = /^[a-z][a-z\d+.-]*:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
  try {
    const parsed = new URL(withScheme);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      return null;
    }
    // "i.ytimg.com." is the same host as "i.ytimg.com" and URL keeps the dot,
    // so host checks would otherwise miss it.
    parsed.hostname = parsed.hostname.toLowerCase().replace(/\.$/, "");
    return parsed;
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

/** An image served by YouTube, such as a video thumbnail. */
export function isYouTubeImageUrl(url: string): boolean {
  const host = parseHttpUrl(url)?.hostname.toLowerCase();
  if (!host) return false;
  return (
    host === "img.youtube.com" ||
    host === "ytimg.com" ||
    host.endsWith(".ytimg.com")
  );
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
  return candidate && VIDEO_ID.test(candidate) && !NOT_VIDEO_IDS.has(candidate)
    ? candidate
    : null;
}

/** Where the link asks playback to start (`t=90`, `t=1m30s`, `start=90`). */
export function getYouTubeStartSeconds(url: string): number | null {
  const parsed = parseYouTubeUrl(url);
  const value =
    parsed?.searchParams.get("t") ?? parsed?.searchParams.get("start");
  if (!value) return null;
  const match = value.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$/);
  if (!match) return null;
  const [, hours = "0", minutes = "0", seconds = "0"] = match;
  const total = Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds);
  return total > 0 ? total : null;
}

// Privacy-enhanced mode: YouTube sets no cookies until the viewer presses play.
export function getYouTubeLinks(url: string) {
  const videoId = getYouTubeVideoId(url);
  if (!videoId) {
    return {
      videoId,
      embedUrl: null,
      watchUrl: parseYouTubeUrl(url)?.toString() ?? null,
    };
  }
  const start = getYouTubeStartSeconds(url);
  const id = encodeURIComponent(videoId);
  const embedUrl = `https://www.youtube-nocookie.com/embed/${id}`;
  const watchUrl = `https://www.youtube.com/watch?v=${id}`;
  return {
    videoId,
    embedUrl: start ? `${embedUrl}?start=${start}` : embedUrl,
    watchUrl: start ? `${watchUrl}&t=${start}s` : watchUrl,
  };
}

export function getYouTubeEmbedUrl(url: string): string | null {
  return getYouTubeLinks(url).embedUrl;
}

export function getYouTubeWatchUrl(url: string): string | null {
  return getYouTubeLinks(url).watchUrl;
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
