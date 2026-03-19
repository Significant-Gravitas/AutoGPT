import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Play } from "@phosphor-icons/react";
import Image from "next/image";
import { useEffect, useState } from "react";
import {
  getYouTubeVideoId,
  isValidVideoFile,
  isValidVideoUrl,
} from "./helpers";
import { useAgentImageItem } from "./useAgentImageItem";

interface AgentImageItemProps {
  image: string;
  index: number;
  playingVideoIndex: number | null;
  handlePlay: (index: number) => void;
  handlePause: (index: number) => void;
}

export function AgentImageItem({
  image,
  index,
  playingVideoIndex,
  handlePlay,
  handlePause,
}: AgentImageItemProps) {
  const { videoRef } = useAgentImageItem({ playingVideoIndex, index });
  const isVideoFile = isValidVideoFile(image);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
  }, [image]);

  return (
    <div className="relative">
      <div className="h-[15rem] overflow-hidden rounded-xl border border-neutral-100 bg-[#a8a8a8] sm:h-[20rem] sm:w-full md:h-[25rem] lg:h-[30rem]">
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
            {!imageLoaded && (
              <Skeleton className="absolute inset-0 rounded-xl" />
            )}
            <Image
              src={image}
              alt="Image"
              fill
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
              className="rounded-xl object-cover"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageLoaded(true)}
            />
          </div>
        )}
      </div>
      {isVideoFile && playingVideoIndex !== index && (
        <div className="absolute bottom-2 left-2 sm:bottom-3 sm:left-3 md:bottom-4 md:left-4 lg:bottom-[1.25rem] lg:left-[1.25rem]">
          <Button
            variant="secondary"
            size="large"
            onClick={() => {
              if (videoRef.current) {
                videoRef.current.play();
              }
            }}
            rightIcon={
              <Play
                size={20}
                weight="fill"
                className="text-black dark:text-neutral-200 sm:h-6 sm:w-6 md:h-7 md:w-7"
              />
            }
          >
            Play demo
          </Button>
        </div>
      )}
    </div>
  );
}
