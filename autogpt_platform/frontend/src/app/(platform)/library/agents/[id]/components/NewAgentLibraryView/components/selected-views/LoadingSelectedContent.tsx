import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { cn } from "@/lib/utils";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../helpers";
import { SelectedViewLayout } from "./SelectedViewLayout";

interface Props {
  agent: LibraryAgent;
  onSelectSettings?: () => void;
  selectedSettings?: boolean;
}

export function LoadingSelectedContent(props: Props) {
  return (
    <SelectedViewLayout
      agent={props.agent}
      onSelectSettings={props.onSelectSettings}
      selectedSettings={props.selectedSettings}
    >
      <div
        className={cn("flex flex-col gap-4", AGENT_LIBRARY_SECTION_PADDING_X)}
      >
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    </SelectedViewLayout>
  );
}
