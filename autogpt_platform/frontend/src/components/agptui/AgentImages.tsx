import * as React from "react";
import Image from "next/image";
import { PlayIcon } from "@radix-ui/react-icons";
import { Button } from "./Button";
interface AgentImagesProps {
  images: string[];
}

const isValidVideoFile = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  return videoExtensions.test(url);
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

const getYouTubeVideoId = (url: string) => {
  const regExp =
    /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
  const match = url.match(regExp);
  return match && match[7].length === 11 ? match[7] : null;
};

export const AgentImages: React.FC<AgentImagesProps> = ({ images }) => {
  const [playingVideoIndex, setPlayingVideoIndex] = React.useState<number | null>(null);

  const handlePlay = (index: number) => {
    setPlayingVideoIndex(index);
  };

  const handlePause = () => {
    setPlayingVideoIndex(null);
  };

  const renderContent = (url: string, index: number) => {
    if (isValidVideoUrl(url)) {
      const videoId = getYouTubeVideoId(url);
      if (videoId) {
        return (
          <iframe
            width="100%"
            height="100%"
            src={`https://www.youtube.com/embed/${videoId}`}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
        );
      } else {
        return (
          <div className="w-full h-full relative overflow-hidden">
            <video
              className="absolute inset-0 w-full h-full object-cover"
              controls
              preload="metadata"
              poster={`${url}#t=0.1`}
              style={{ objectPosition: 'center 25%' }}
              onPlay={() => handlePlay(index)}
              onPause={handlePause}
              autoPlay={false}
            >
              <source src={url} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        );
      }
    } else if (isValidImageUrl(url)) {
      return (
        <Image
          src={url}
          alt="Image"
          layout="fill"
          objectFit="cover"
          className="rounded-xl"
        />
      );
    } else {
      return <div className="w-full h-full flex items-center justify-center">Unsupported content</div>;
    }
  };

  return (
    <div className="w-[56.25rem] h-[91.25rem] overflow-y-auto">
      <div className="space-y-[1.875rem]">
        {images.map((image, index) => (
          <div key={index} className="relative">
            <div className="w-full h-[30rem] bg-[#a8a8a8] rounded-xl overflow-hidden">
              {renderContent(image, index)}
            </div>
            {isValidVideoFile(image) && playingVideoIndex !== index && (
              <div className="absolute bottom-[1.25rem] left-[1.25rem]">
                <Button
                  variant="default"
                  size="default"
                  onClick={() => {
                    console.log('video index', index);
                    const videos = document.querySelectorAll('video');
                    if (playingVideoIndex !== null) {
                      videos[playingVideoIndex]?.pause();
                    }
                    const currentVideo = videos[index];
                    if (currentVideo) {
                      if (playingVideoIndex === index) {
                        currentVideo.pause();
                        setPlayingVideoIndex(null);
                      } else {
                        currentVideo.play();
                        handlePlay(index);
                      }
                    }
                  }}
                >
                  <span className="text-[#272727] text-xl font-medium font-neue leading-9 tracking-tight pr-2">
                    Play demo
                  </span>
                  <PlayIcon className="w-7 h-7 text-black" />
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
