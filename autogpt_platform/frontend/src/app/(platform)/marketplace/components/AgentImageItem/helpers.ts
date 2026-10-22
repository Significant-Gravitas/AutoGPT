import { isYouTubeUrl } from "@/components/molecules/YouTubeEmbed/helpers";

export const isValidVideoFile = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  return videoExtensions.test(url);
};

export const isValidVideoUrl = (url: string): boolean => {
  return isValidVideoFile(url) || isYouTubeUrl(url);
};
