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
  if (url.startsWith("data:video")) {
    return true;
  }
  const validUrl = /^(https?:\/\/)(www\.)?/i;
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  const cleanedUrl = url.split("?")[0];
  return (
    (validUrl.test(cleanedUrl) && videoExtensions.test(cleanedUrl)) ||
    youtubeRegex.test(cleanedUrl)
  );
};

const isValidImageUrl = (url: string): boolean => {
  if (url.startsWith("data:image/")) {
    return true;
  }
  const imageExtensions = /\.(jpeg|jpg|gif|png|svg|webp)$/i;
  const cleanedUrl = url.split("?")[0];
  return imageExtensions.test(cleanedUrl);
};

const isValidAudioUrl = (url: string): boolean => {
  if (url.startsWith("data:audio")) {
    return true;
  }
  const validUrl = /^(https?:\/\/)(www\.)?/i;
  const audioExtensions = /\.(mp3|wav)$/i;
  const cleanedUrl = url.split("?")[0];
  return validUrl.test(cleanedUrl) && audioExtensions.test(cleanedUrl);
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
