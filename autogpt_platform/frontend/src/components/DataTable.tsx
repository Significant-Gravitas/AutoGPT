import React from 'react';
import { beautifyString } from "@/lib/utils";
import { Button } from "./ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Clipboard } from "lucide-react";
import { useToast } from "./ui/use-toast";

type DataTableProps = {
  title?: string;
  truncateLongData?: boolean;
  data: { [key: string]: Array<any> };
};

const VideoRenderer: React.FC<{ videoUrl: string }> = ({ videoUrl }) => {
  const getYouTubeVideoId = (url: string) => {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[7].length === 11) ? match[7] : null;
  };

  const videoId = getYouTubeVideoId(videoUrl);

  return (
    <div className="w-full p-2">
      {videoId ? (
        <iframe
          width="100%"
          height="315"
          src={`https://www.youtube.com/embed/${videoId}`}
          frameBorder="0"
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
    <img src={imageUrl} alt="Image" className="max-w-full h-auto" />
  </div>
);

const isValidVideoUrl = (url: string): boolean => {
  const videoExtensions = /\.(mp4|webm|ogg)$/i;
  const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$/;
  return videoExtensions.test(url) || youtubeRegex.test(url);
};

const isValidImageUrl = (url: string): boolean => {
  const imageExtensions = /\.(jpeg|jpg|gif|png|svg|webp)$/i;
  return imageExtensions.test(url);
};

export default function DataTable({
  title,
  truncateLongData,
  data,
}: DataTableProps) {
  const { toast } = useToast();
  const maxChars = 100;

  const copyData = (pin: string, data: string) => {
    navigator.clipboard.writeText(data).then(() => {
      toast({
        title: `"${pin}" data copied to clipboard!`,
        duration: 2000,
      });
    });
  };

  const renderCellContent = (value: any) => {
    if (typeof value === "string") {
      if (isValidVideoUrl(value)) {
        return <VideoRenderer videoUrl={value} />;
      } else if (isValidImageUrl(value)) {
        return <ImageRenderer imageUrl={value} />;
      }
    }

    const text =
      typeof value === "object" ? JSON.stringify(value) : String(value);
    return truncateLongData && text.length > maxChars
      ? text.slice(0, maxChars) + "..."
      : text;
  };

  return (
    <>
      {title && <strong className="mt-2 flex justify-center">{title}</strong>}
      <Table className="cursor-default select-text">
        <TableHeader>
          <TableRow>
            <TableHead>Pin</TableHead>
            <TableHead>Data</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(data).map(([key, value]) => (
            <TableRow className="group" key={key}>
              <TableCell className="cursor-text">
                {beautifyString(key)}
              </TableCell>
              <TableCell className="cursor-text">
                <div className="flex min-h-9 items-center">
                  <Button
                    className="absolute right-1 top-auto m-1 hidden p-2 group-hover:block"
                    variant="outline"
                    size="icon"
                    onClick={() =>
                      copyData(
                        beautifyString(key),
                        value
                          .map((i) =>
                            typeof i === "object" ? JSON.stringify(i) : String(i)
                          )
                          .join(", ")
                      )
                    }
                    title="Copy Data"
                  >
                    <Clipboard size={18} />
                  </Button>
                  {value.map((item, index) => (
                    <React.Fragment key={index}>
                      {renderCellContent(item)}
                      {index < value.length - 1 && ", "}
                    </React.Fragment>
                  ))}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}
