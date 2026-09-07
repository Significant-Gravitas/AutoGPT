import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowLeft01Icon, Folder02Icon } from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";
import { TEAM_GRID_CLASS } from "../../helpers";

interface Props {
  isLoading: boolean;
  podGroups: { pod: ExpertPod; experts: Expert[] }[];
  onNewPod: () => void;
  renderCard: (expert: Expert) => ReactNode;
}

const FOLDER_GRID_CLASS =
  "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3";

export function PodBoard({
  isLoading,
  podGroups,
  onNewPod,
  renderCard,
}: Props) {
  const [openPodId, setOpenPodId] = useState<string | null>(null);
  const openGroup = podGroups.find((group) => group.pod.id === openPodId);

  if (openGroup) {
    return (
      <section
        aria-label={`${openGroup.pod.name} pod`}
        className="flex flex-col gap-4"
      >
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            leadingIcon={ArrowLeft01Icon}
            aria-label="Back to pods"
            onClick={() => setOpenPodId(null)}
          />
          <Text variant="large-medium" as="h5" tone="primary">
            {openGroup.pod.name}
          </Text>
          <Text variant="small" tone="muted">
            {openGroup.experts.length}{" "}
            {openGroup.experts.length === 1 ? "expert" : "experts"}
          </Text>
        </div>
        {openGroup.experts.length > 0 ? (
          <div className={TEAM_GRID_CLASS}>
            {openGroup.experts.map(renderCard)}
          </div>
        ) : (
          <Text variant="body" tone="muted">
            No experts in this pod yet. Move one in from its card menu.
          </Text>
        )}
      </section>
    );
  }

  return (
    <section aria-label="Pods" className="flex flex-col gap-4">
      <Text variant="large-medium" as="h5" tone="primary">
        Pods
      </Text>
      {isLoading ? (
        <div className={FOLDER_GRID_CLASS}>
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} className="h-28 w-full rounded-xl" />
          ))}
        </div>
      ) : podGroups.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-zinc-300 bg-white p-8 text-center">
          <Text variant="large-medium" tone="primary">
            No pods yet
          </Text>
          <Text variant="body" tone="secondary" className="max-w-prose">
            Group experts into pods to keep related work together.
          </Text>
          <Button variant="primary" size="xs" onClick={onNewPod}>
            New Pod
          </Button>
        </div>
      ) : (
        <div className={FOLDER_GRID_CLASS}>
          {podGroups.map((group) => (
            <PodFolder
              key={group.pod.id}
              name={group.pod.name}
              experts={group.experts}
              onOpen={() => setOpenPodId(group.pod.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

interface PodFolderProps {
  name: string;
  experts: Expert[];
  onOpen: () => void;
}

function PodFolder({ name, experts, onOpen }: PodFolderProps) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`Open ${name} pod`}
      className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-white p-4 text-left transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
    >
      <div className="flex items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-zinc-100 text-zinc-600">
          <Icon icon={Folder02Icon} size={20} />
        </span>
        <div className="min-w-0">
          <Text
            variant="large-medium"
            as="h5"
            tone="primary"
            className="truncate"
          >
            {name}
          </Text>
          <Text variant="small" tone="muted">
            {experts.length} {experts.length === 1 ? "expert" : "experts"}
          </Text>
        </div>
      </div>
      {experts.length > 0 ? (
        <ul className="flex flex-wrap gap-1.5">
          {experts.map((expert) => (
            <Text
              key={expert.id}
              variant="small"
              as="li"
              tone="secondary"
              className="max-w-full truncate rounded-md bg-zinc-50 px-2 py-0.5 ring-1 ring-inset ring-zinc-200"
            >
              {expert.name}
            </Text>
          ))}
        </ul>
      ) : (
        <Text variant="small" tone="muted">
          No experts in this pod yet.
        </Text>
      )}
    </button>
  );
}
