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
    const [isVideoPlaying, setIsVideoPlaying] = React.useState(false);
    const [thumbnail, setThumbnail] = React.useState<string>("");

    React.useEffect(() => {
      if (
        playingVideoIndex !== index &&
        videoRef.current &&
        !videoRef.current.paused
      ) {
        videoRef.current.pause();
      }
    }, [playingVideoIndex, index]);

    React.useEffect(() => {
      if (videoRef.current && isValidVideoFile(image)) {
        videoRef.current.currentTime = 0.1;
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0);
        setThumbnail(canvas.toDataURL());
      }
    }, [image]);

    const isVideoFile = isValidVideoFile(image);

    return (
      <div className="relative">
        <div className="h-[15rem] overflow-hidden rounded-[26px] bg-[#a8a8a8] dark:bg-neutral-700 sm:h-[20rem] sm:w-full md:h-[25rem] lg:h-[30rem]">
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
                  controls={isVideoPlaying}
                  preload="metadata"
                  poster={thumbnail || `${image}#t=0.1`}
                  style={{ objectPosition: "center 25%" }}
                  onPlay={() => {
                    setIsVideoPlaying(true);
                    handlePlay(index);
                  }}
                  onPause={() => {
                    setIsVideoPlaying(false);
                    handlePause(index);
                  }}
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
          <div className="absolute bottom-2 left-2 sm:bottom-3 sm:left-3 md:bottom-4 md:left-4 lg:bottom-[1.25rem] lg:left-[1.25rem]">
            <Button
              size="default"
              className="space-x-2 border border-zinc-300 bg-zinc-400/50 backdrop-blur-xl hover:bg-zinc-400/80 sm:h-14"
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.play();
                }
              }}
            >
              <PlayIcon className="h-5 w-5 text-white dark:text-neutral-200 sm:h-6 sm:w-6 md:h-7 md:w-7" />

              <span className="font-poppins text-sm font-medium text-white dark:text-neutral-200 sm:text-lg">
                Play demo
              </span>
            </Button>
          </div>
        )}
      </div>
    );
  },
);

AgentImageItem.displayName = "AgentImageItem";
