import type { GalleryBlock } from "@/app/api/__generated__/models/galleryBlock";
import { Text } from "@/components/atoms/Text/Text";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { defaultWorkspaceFileUrl } from "../../../components/ChatMessagesContainer/helpers";
import { BlockCard } from "./BlockCard";

interface Props {
  block: GalleryBlock;
  fileUrlBuilder?: (fileId: string) => string;
}

export function GalleryBlockView({
  block,
  fileUrlBuilder = defaultWorkspaceFileUrl,
}: Props) {
  return (
    <BlockCard title={block.title}>
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {block.images.map((image, index) => {
          const file = parseWorkspaceURI(image.src);
          if (!file) return null;
          return (
            <li key={index} className="min-w-0">
              {/* eslint-disable-next-line @next/next/no-img-element -- a user or workspace image URL next/image cannot optimise */}
              <img
                src={fileUrlBuilder(file.fileID)}
                alt={image.caption}
                loading="lazy"
                className="aspect-square w-full rounded-xl object-cover smooth-shadow-ring-sm"
              />
              <Text
                variant="small"
                tone="secondary"
                className="mt-2 break-words"
              >
                {image.caption}
              </Text>
            </li>
          );
        })}
      </ul>
    </BlockCard>
  );
}
