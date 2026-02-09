"use client";

import * as React from "react";

const getYouTubeVideoId = (url: string) => {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
};

const isValidMediaUri = (url: string): boolean => {
  if (url.startsWith("data:")) {
    return true;
  }
  const validUrl = /^(https?:\/\/)(www\.)?/i;
  const cleanedUrl = url.split("?")[0];
  return validUrl.test(cleanedUrl);
};

const isValidVideoUrl = (url: string): boolean => {
  if (url.startsWith("data:video")) {
    return true;
  }
  const videoExtensions = /\.(mp4|webm|ogg|mov|avi|mkv|m4v)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  const cleanedUrl = url.split("?")[0];
  return (
    (isValidMediaUri(url) && videoExtensions.test(cleanedUrl)) ||
    youtubeRegex.test(cleanedUrl)
  );
};

const isValidImageUrl = (url: string): boolean => {
  if (url.startsWith("data:image/")) {
    return true;
  }
  const imageExtensions = /\.(jpeg|jpg|gif|png|svg|webp)$/i;
  const cleanedUrl = url.split("?")[0];
  return isValidMediaUri(url) && imageExtensions.test(cleanedUrl);
};

const isValidAudioUrl = (url: string): boolean => {
  if (url.startsWith("data:audio")) {
    return true;
  }
  const audioExtensions = /\.(mp3|wav|ogg|m4a|aac|flac)$/i;
  const cleanedUrl = url.split("?")[0];
  return isValidMediaUri(url) && audioExtensions.test(cleanedUrl);
};

const getVideoMimeType = (url: string): string => {
  if (url.startsWith("data:video/")) {
    const match = url.match(/^data:(video\/[^;]+)/);
    return match?.[1] || "video/mp4";
  }
  const extension = url.split("?")[0].split(".").pop()?.toLowerCase();
  const mimeMap: Record<string, string> = {
    mp4: "video/mp4",
    webm: "video/webm",
    ogg: "video/ogg",
    mov: "video/quicktime",
    avi: "video/x-msvideo",
    mkv: "video/x-matroska",
    m4v: "video/mp4",
  };
  return mimeMap[extension || ""] || "video/mp4";
};

const VideoRenderer: React.FC<{ videoUrl: string }> = ({ videoUrl }) => {
  const videoId = getYouTubeVideoId(videoUrl);
  return (
    <div className="w-full p-2">
      {videoId ? (
        <iframe
          width="100%"
          height="315"
          src={`https://www.youtube.com/embed/${videoId}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>
      ) : (
        <video controls width="100%" height="315">
          <source src={videoUrl} type={getVideoMimeType(videoUrl)} />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

const ImageRenderer: React.FC<{ imageUrl: string }> = ({ imageUrl }) => {
  return (
    <div className="w-full p-2">
      <picture>
        <img
          src={imageUrl}
          alt="Image"
          className="h-auto max-w-full"
          width="100%"
          height="auto"
        />
      </picture>
    </div>
  );
};

const AudioRenderer: React.FC<{ audioUrl: string }> = ({ audioUrl }) => (
  <div className="w-full p-2">
    <audio controls className="w-full">
      <source
        src={audioUrl}
        type={`audio/${audioUrl.split(".").pop()?.toLowerCase()}`}
      />
      Your browser does not support the audio element.
    </audio>
  </div>
);

export const TextRenderer: React.FC<{
  value: any;
  truncateLengthLimit?: number;
}> = ({ value, truncateLengthLimit }) => {
  const text =
    typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  return truncateLengthLimit && text.length > truncateLengthLimit
    ? text.slice(0, truncateLengthLimit) + "..."
    : text;
};

export const ContentRenderer: React.FC<{
  value: any;
  truncateLongData?: boolean;
}> = ({ value, truncateLongData }) => {
  if (typeof value === "string") {
    if (value.startsWith("data:image/")) {
      return <ImageRenderer imageUrl={value} />;
    }
    if (isValidVideoUrl(value)) {
      return <VideoRenderer videoUrl={value} />;
    } else if (isValidImageUrl(value)) {
      return <ImageRenderer imageUrl={value} />;
    } else if (isValidAudioUrl(value)) {
      return <AudioRenderer audioUrl={value} />;
    }
  }
  return (
    <TextRenderer
      value={value}
      truncateLengthLimit={truncateLongData ? 100 : undefined}
    />
  );
};
