import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Folder02Icon } from "@hugeicons/core-free-icons";

interface Props {
  isLoading: boolean;
  podGroups: { pod: ExpertPod; experts: Expert[] }[];
  ungroupedExperts: Expert[];
  onNewPod: () => void;
}

const FOLDER_GRID_CLASS =
  "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3";

export function PodBoard({
  isLoading,
  podGroups,
  ungroupedExperts,
  onNewPod,
}: Props) {
  if (isLoading) {
    return (
      <div className={FOLDER_GRID_CLASS}>
        {[0, 1, 2].map((index) => (
          <Skeleton key={index} className="h-28 w-full rounded-2xl" />
        ))}
      </div>
    );
  }

  if (podGroups.length === 0 && ungroupedExperts.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-[1.75rem] border border-dashed border-zinc-300 bg-white p-10 text-center">
        <Text variant="large-medium">No pods yet</Text>
        <Text variant="body" className="max-w-prose text-zinc-600">
          Group experts into pods to keep related work together.
        </Text>
        <Button variant="primary" size="small" onClick={onNewPod}>
          New Pod
        </Button>
      </div>
    );
  }

  return (
    <div className={FOLDER_GRID_CLASS}>
      {podGroups.map((group) => (
        <PodFolder
          key={group.pod.id}
          name={group.pod.name}
          experts={group.experts}
        />
      ))}
      {ungroupedExperts.length > 0 ? (
        <PodFolder name="Ungrouped" experts={ungroupedExperts} />
      ) : null}
    </div>
  );
}

interface PodFolderProps {
  name: string;
  experts: Expert[];
}

function PodFolder({ name, experts }: PodFolderProps) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl bg-white p-4 shadow-zinc-950 smooth-shadow-ring-sm">
      <div className="flex items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-zinc-100 text-zinc-600">
          <Icon icon={Folder02Icon} size={20} />
        </span>
        <div className="min-w-0">
          <Text variant="h5" className="truncate">
            {name}
          </Text>
          <Text variant="small" className="text-zinc-500">
            {experts.length} {experts.length === 1 ? "expert" : "experts"}
          </Text>
        </div>
      </div>
      {experts.length > 0 ? (
        <ul className="flex flex-wrap gap-1.5">
          {experts.map((expert) => (
            <li
              key={expert.id}
              className="max-w-full truncate rounded-full bg-zinc-50 px-2.5 py-1 text-xs text-zinc-600 ring-1 ring-inset ring-zinc-200"
            >
              {expert.name}
            </li>
          ))}
        </ul>
      ) : (
        <Text variant="small" className="text-zinc-500">
          No experts in this pod yet.
        </Text>
      )}
    </section>
  );
}
