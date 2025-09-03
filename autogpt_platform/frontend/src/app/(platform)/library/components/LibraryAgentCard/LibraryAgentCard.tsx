import Link from "next/link";
import Image from "next/image";
import { Heart } from "@phosphor-icons/react";
import { useState } from "react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import BackendAPI from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";

interface LibraryAgentCardProps {
  agent: LibraryAgent;
}

export default function LibraryAgentCard({
  agent: {
    id,
    name,
    description,
    graph_id,
    can_access_graph,
    creator_image_url,
    image_url,
    is_favorite,
  },
}: LibraryAgentCardProps) {
  const [isFavorite, setIsFavorite] = useState(is_favorite);
  const [isUpdating, setIsUpdating] = useState(false);
  const { toast } = useToast();
  const api = new BackendAPI();

  const handleToggleFavorite = async (e: React.MouseEvent) => {
    e.preventDefault(); // Prevent navigation when clicking the heart
    e.stopPropagation();
    
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      await api.updateLibraryAgent(id, { is_favorite: !isFavorite });
      setIsFavorite(!isFavorite);
      toast({
        title: isFavorite ? "Removed from favorites" : "Added to favorites",
        description: `${name} has been ${isFavorite ? "removed from" : "added to"} your favorites.`,
      });
    } catch (error) {
      console.error("Failed to update favorite status:", error);
      toast({
        title: "Error",
        description: "Failed to update favorite status. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div
      data-testid="library-agent-card"
      data-agent-id={id}
      className="inline-flex w-full max-w-[434px] flex-col items-start justify-start gap-2.5 rounded-[26px] bg-white transition-all duration-300 hover:shadow-lg dark:bg-transparent dark:hover:shadow-gray-700"
    >
      <Link
        href={`/library/agents/${id}`}
        className="relative h-[200px] w-full overflow-hidden rounded-[20px]"
      >
        {!image_url ? (
          <div
            className={`h-full w-full ${
              [
                "bg-gradient-to-r from-green-200 to-blue-200",
                "bg-gradient-to-r from-pink-200 to-purple-200",
                "bg-gradient-to-r from-yellow-200 to-orange-200",
                "bg-gradient-to-r from-blue-200 to-cyan-200",
                "bg-gradient-to-r from-indigo-200 to-purple-200",
              ][parseInt(id.slice(0, 8), 16) % 5]
            }`}
            style={{
              backgroundSize: "200% 200%",
              animation: "gradient 15s ease infinite",
            }}
          />
        ) : (
          <Image
            src={image_url}
            alt={`${name} preview image`}
            fill
            className="object-cover"
            priority
          />
        )}
        <button
          onClick={handleToggleFavorite}
          className={cn(
            "absolute right-4 top-4 p-2 rounded-full bg-white/90 backdrop-blur-sm transition-all duration-200",
            "hover:scale-110 hover:bg-white",
            "focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2",
            isUpdating && "opacity-50 cursor-not-allowed"
          )}
          disabled={isUpdating}
          aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
        >
          <Heart 
            size={20} 
            weight={isFavorite ? "fill" : "regular"}
            className={cn(
              "transition-colors duration-200",
              isFavorite ? "text-red-500" : "text-gray-600 hover:text-red-500"
            )}
          />
        </button>
        <div className="absolute bottom-4 left-4">
          <Avatar className="h-16 w-16">
            <AvatarImage
              src={
                creator_image_url
                  ? creator_image_url
                  : "/avatar-placeholder.png"
              }
              alt={`${name} creator avatar`}
            />
            <AvatarFallback size={64}>{name.charAt(0)}</AvatarFallback>
          </Avatar>
        </div>
      </Link>

      <div className="flex w-full flex-1 flex-col px-4 py-4">
        <Link href={`/library/agents/${id}`}>
          <h3 className="mb-2 line-clamp-2 font-poppins text-2xl font-semibold leading-tight text-[#272727] dark:text-neutral-100">
            {name}
          </h3>

          <p className="line-clamp-3 flex-1 text-sm text-gray-600 dark:text-gray-400">
            {description}
          </p>
        </Link>

        <div className="flex-grow" />
        {/* Spacer */}

        <div className="items-between mt-4 flex w-full justify-between gap-3">
          <Link
            href={`/library/agents/${id}`}
            className="text-lg font-semibold text-neutral-800 hover:underline dark:text-neutral-200"
          >
            See runs
          </Link>

          {can_access_graph && (
            <Link
              href={`/build?flowID=${graph_id}`}
              className="text-lg font-semibold text-neutral-800 hover:underline dark:text-neutral-200"
            >
              Open in builder
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
