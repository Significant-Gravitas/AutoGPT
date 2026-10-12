import {
  useGetV2ListBoards,
  usePostV2CreateBoard,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { BOARDS_LIST_QUERY_KEY } from "@/app/(platform)/home/board-helpers";
import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { useJumpToNavGroup } from "../../useJumpToNavGroup";
import {
  errorDetail,
  isBoardPath,
  MAX_BOARDS,
  newProjectHref,
  nextProjectName,
} from "./helpers";

export function useProjectsNavSection() {
  const router = useRouter();
  const pathname = usePathname();
  const queryClient = useQueryClient();
  const jump = useJumpToNavGroup();

  const { data: boards = [] } = useGetV2ListBoards({
    query: {
      queryKey: BOARDS_LIST_QUERY_KEY,
      select: (res) => (res.status === 200 ? res.data.boards : []),
    },
  });
  const projects = boards.filter((board) => !board.is_default);
  const isAtLimit = boards.length >= MAX_BOARDS;

  const { mutate: createBoard, isPending: isCreating } = usePostV2CreateBoard({
    mutation: {
      onSuccess: (res) => {
        void queryClient.invalidateQueries({ queryKey: BOARDS_LIST_QUERY_KEY });
        if (res.status === 200) router.push(newProjectHref(res.data.id));
      },
      onError: (error) => {
        toast({
          title: "Couldn't create the project",
          description: errorDetail(error),
          variant: "destructive",
        });
      },
    },
  });

  function createProject() {
    if (isAtLimit || isCreating) return;
    const name = nextProjectName(boards.map((board) => board.name));
    createBoard({ data: { name } });
  }

  return {
    projects,
    isAtLimit,
    isCreating,
    isOnProject: isBoardPath(pathname),
    isProjectActive: (boardId: string) => isBoardPath(pathname, boardId),
    createProject,
    ...jump,
  };
}
