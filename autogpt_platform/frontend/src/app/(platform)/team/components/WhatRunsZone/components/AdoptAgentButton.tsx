import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { useState } from "react";

interface Props {
  agent: LibraryAgent;
  experts: Expert[];
  isPending: boolean;
  onAdopt: (agent: LibraryAgent, expert: Expert) => void;
}

export function AdoptAgentButton({
  agent,
  experts,
  isPending,
  onAdopt,
}: Props) {
  const [open, setOpen] = useState(false);

  function handleSelect(expert: Expert) {
    setOpen(false);
    onAdopt(agent, expert);
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="secondary" size="small" loading={isPending}>
          Adopt
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 p-2">
        <Text variant="small-medium" className="px-2 pb-1 pt-1 text-zinc-500">
          Adopt into a thread
        </Text>
        <div className="flex flex-col">
          {experts.map((expert) => (
            <button
              key={expert.id}
              type="button"
              onClick={() => handleSelect(expert)}
              className="flex items-start gap-2 rounded-lg px-2 py-2 text-left hover:bg-zinc-50"
            >
              <ExpertAvatar
                name={expert.name}
                avatarUrl={expert.avatar_url ?? null}
                size={28}
              />
              <div className="min-w-0 flex-1">
                <Text variant="body" className="truncate">
                  {expert.name}
                </Text>
                <Text variant="small" className="text-zinc-500">
                  Runs will show up in {expert.name}&apos;s thread. You can undo
                  anytime.
                </Text>
              </div>
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
