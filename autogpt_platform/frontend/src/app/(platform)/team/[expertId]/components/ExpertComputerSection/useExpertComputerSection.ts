import {
  getGetExpertComputerQueryKey,
  useGetExpertComputer,
  useStartExpertDesktop,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { DesktopStream } from "@/app/api/__generated__/models/desktopStream";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  expertId: string;
  enabled: boolean;
}

export function useExpertComputerSection({ expertId, enabled }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [stream, setStream] = useState<DesktopStream | null>(null);

  const computerQuery = useGetExpertComputer(expertId, {
    query: {
      select: (res) => okData(res) ?? null,
      enabled,
      refetchInterval: 15_000,
    },
  });

  const { mutate: startDesktop, isPending: isOpening } = useStartExpertDesktop({
    mutation: {
      onSuccess: (res) => {
        const next = okData(res);
        if (next) setStream(next);
        queryClient.invalidateQueries({
          queryKey: getGetExpertComputerQueryKey(expertId),
        });
      },
      onError: () => {
        toast({
          title: "Could not open the desktop",
          description: "The sandbox did not come up. Try again in a moment.",
          variant: "destructive",
        });
      },
    },
  });

  function openDesktop() {
    startDesktop({ expertId });
  }

  return {
    computer: computerQuery.data ?? null,
    isLoading: computerQuery.isLoading,
    isError: computerQuery.isError,
    refetch: () => computerQuery.refetch(),
    stream,
    openDesktop,
    isOpening,
  };
}
