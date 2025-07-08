import { useRouter } from "next/navigation";

// TODO : Need to add more logic for pagination in future
export const useAgentsSection = () => {
  const router = useRouter();

  const handleCardClick = (creator: string, slug: string) => {
    router.push(
      `/marketplace/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`,
    );
  };

  return { handleCardClick };
};
