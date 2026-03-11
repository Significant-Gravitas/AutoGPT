import * as React from "react";
import Image from "next/image";
import { PlayIcon } from "@radix-ui/react-icons";
import { Button } from "./Button";

const isValidVideoFile = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  return videoExtensions.test(url);
};

const isValidVideoUrl = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  return videoExtensions.test(url) || youtubeRegex.test(url);
};

const getYouTubeVideoId = (url: string) => {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
};

interface AgentImageItemProps {
  image: string;
  index: number;
  playingVideoIndex: number | null;
  handlePlay: (index: number) => void;
  handlePause: (index: number) => void;
}

export const AgentImageItem: React.FC<AgentImageItemProps> = React.memo(
  ({ image, index, playingVideoIndex, handlePlay, handlePause }) => {
    const videoRef = React.useRef<HTMLVideoElement>(null);

    React.useEffect(() => {
      if (
        playingVideoIndex !== index &&
        videoRef.current &&
        !videoRef.current.paused
      ) {
        videoRef.current.pause();
      }
    }, [playingVideoIndex, index]);

    const isVideoFile = isValidVideoFile(image);

    return (
      <div className="relative">
        <div className="h-60 overflow-hidden rounded-[26px] bg-customGray-200 dark:bg-neutral-700 sm:h-80 sm:w-full md:h-100 lg:h-120">
          {isValidVideoUrl(image) ? (
            getYouTubeVideoId(image) ? (
              <iframe
                width="100%"
                height="100%"
                src={`https://www.youtube.com/embed/${getYouTubeVideoId(image)}`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title="YouTube video player"
              ></iframe>
            ) : (
              <div className="relative h-full w-full overflow-hidden">
                <video
                  ref={videoRef}
                  className="absolute inset-0 h-full w-full object-cover"
                  controls
                  preload="metadata"
                  poster={`${image}#t=0.1`}
                  style={{ objectPosition: "center 25%" }}
                  onPlay={() => handlePlay(index)}
                  onPause={() => handlePause(index)}
                  autoPlay={false}
                  title="Video"
                >
                  <source src={image} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
            )
          ) : (
            <div className="relative h-full w-full">
              <Image
                src={image}
                alt="Image"
                fill
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                className="rounded-xl object-cover"
              />
            </div>
          )}
        </div>
        {isVideoFile && playingVideoIndex !== index && (
          <div className="absolute bottom-2 left-2 sm:bottom-3 sm:left-3 md:bottom-4 md:left-4 lg:bottom-5 lg:left-5">
            <Button
              size="default"
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.play();
                }
              }}
            >
              <span className="pr-1 text-sm font-medium leading-6 tracking-tight text-customGray-700 dark:text-neutral-200 sm:pr-2 sm:text-base sm:leading-7 md:text-lg md:leading-8 lg:text-xl lg:leading-9">
                Play demo
              </span>
              <PlayIcon className="h-5 w-5 text-black dark:text-neutral-200 sm:h-6 sm:w-6 md:h-7 md:w-7" />
            </Button>
          </div>
        )}
      </div>
    );
  },
);

AgentImageItem.displayName = "AgentImageItem";
