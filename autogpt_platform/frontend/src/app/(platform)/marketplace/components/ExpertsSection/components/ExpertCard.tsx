import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  expert: Expert;
  isHired: boolean;
  onClick: () => void;
}

export function ExpertCard({ expert, isHired, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-5 text-left transition-shadow hover:shadow-md"
    >
      <div className="flex items-center gap-3">
        <Avatar className="h-12 w-12">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <Text variant="large-medium">{expert.name}</Text>
          <Text variant="small" className="text-zinc-500">
            {expert.role}
          </Text>
        </div>
        {isHired ? <Badge variant="success">Hired</Badge> : null}
      </div>
      {expert.tagline ? <Text variant="body">{expert.tagline}</Text> : null}
      <Text variant="small" className="text-zinc-500">
        {expert.workflows.length} preloaded workflows
      </Text>
    </button>
  );
}
