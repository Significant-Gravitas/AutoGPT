import {
  getYouTubeVideoId,
  isYouTubeImageUrl,
} from "@/components/molecules/YouTubeEmbed/helpers";
import { isValidVideoUrl } from "../AgentImageItem/helpers";

type Thumbnail =
  | { kind: "image"; src: string }
  | { kind: "youtube"; label: string }
  | { kind: "video" };

// Anything served by YouTube (a video's thumbnail, or a listing image hosted
// there) waits for the same marketing consent as the player.
export function getThumbnail(
  image: string,
  hasAdvertisingConsent: boolean,
): Thumbnail {
  if (isValidVideoUrl(image)) {
    const videoId = getYouTubeVideoId(image);
    if (!videoId) return { kind: "video" };
    if (!hasAdvertisingConsent) {
      return { kind: "youtube", label: "YouTube video" };
    }
    return {
      kind: "image",
      src: `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`,
    };
  }
  if (isYouTubeImageUrl(image) && !hasAdvertisingConsent) {
    return { kind: "youtube", label: "Image from YouTube" };
  }
  return { kind: "image", src: image };
}
