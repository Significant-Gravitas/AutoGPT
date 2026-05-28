import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  useGetV2GetLibraryAgentByGraphId,
  usePostV2ForkLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useRouter } from "next/navigation";
import { parseAsString, useQueryStates } from "nuqs";

export function useIsReadOnlyGraph() {
  const router = useRouter();
  const { toast } = useToast();
  const { user } = useSupabase();

  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
      },
    },
  );

  // Treat the loading state as not read-only to avoid flickering the banner
  // during initial mount; the canvas already shows a loading box at that stage.
  const isReadOnly = !!graph && !!user && graph.user_id !== user.id;

  const { data: libraryAgent } = useGetV2GetLibraryAgentByGraphId(
    flowID ?? "",
    {},
    {
      query: {
        select: (res) => res.data as LibraryAgent,
        enabled: !!flowID && isReadOnly,
      },
    },
  );

  const { mutateAsync: forkAgent, isPending: isDuplicating } =
    usePostV2ForkLibraryAgent();

  async function duplicate() {
    if (!libraryAgent) return;

    try {
      const result = await forkAgent({ libraryAgentId: libraryAgent.id });
      const forked = result.data as LibraryAgent;
      router.push(`/build?flowID=${forked.graph_id}`);
    } catch (error: unknown) {
      toast({
        title: "Failed to duplicate agent",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    isReadOnly,
    duplicate,
    isDuplicating,
    canDuplicate: !!libraryAgent,
  };
}
