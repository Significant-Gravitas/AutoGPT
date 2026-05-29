export const isValidVideoFile = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  return videoExtensions.test(url);
};

export const isValidVideoUrl = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  return videoExtensions.test(url) || youtubeRegex.test(url);
};

export const getYouTubeVideoId = (url: string) => {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
};
