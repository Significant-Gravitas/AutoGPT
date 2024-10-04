"use client";

import * as React from "react";
import Image from "next/image";

const getYouTubeVideoId = (url: string) => {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
};

const isValidVideoUrl = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  return videoExtensions.test(url) || youtubeRegex.test(url);
};

const isValidImageUrl = (url: string): boolean => {
  const imageExtensions = /\.(jpeg|jpg|gif|png|svg|webp)$/i;
  const cleanedUrl = url.split("?")[0];
  return imageExtensions.test(cleanedUrl);
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
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};

const ImageRenderer: React.FC<{ imageUrl: string }> = ({ imageUrl }) => (
  <div className="w-full p-2">
    <img
      src={imageUrl}
      alt="Image"
      className="h-auto max-w-full"
      width="100%"
      height="auto"
    />
  </div>
);

const TextRenderer: React.FC<{ value: any; truncateLongData?: boolean }> = ({
  value,
  truncateLongData,
}) => {
  const maxChars = 100;
  const text =
    typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  return truncateLongData && text.length > maxChars
    ? text.slice(0, maxChars) + "..."
    : text;
};

export const ContentRenderer: React.FC<{
  value: any;
  truncateLongData?: boolean;
}> = ({ value, truncateLongData }) => {
  if (typeof value === "string") {
    if (isValidVideoUrl(value)) {
      return <VideoRenderer videoUrl={value} />;
    } else if (isValidImageUrl(value)) {
      return <ImageRenderer imageUrl={value} />;
    }
  }
  return <TextRenderer value={value} truncateLongData={truncateLongData} />;
};
