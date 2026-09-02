import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { OfficeTemplate } from "../api";

interface Props {
  template: OfficeTemplate;
  onSelect: (templateId: string) => void;
}

export function OfficePackCard({ template, onSelect }: Props) {
  const count = template.experts.length;

  return (
    <button
      type="button"
      onClick={() => onSelect(template.id)}
      className="group flex flex-col items-start gap-2 rounded-3xl border border-zinc-200 bg-white p-5 text-left transition-colors hover:border-zinc-300 hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
    >
      <Text variant="large-medium" className="text-zinc-900">
        {template.name}
      </Text>
      <Text variant="body" className="line-clamp-2 text-zinc-600">
        {template.description}
      </Text>
      <span className="mt-auto inline-flex items-center gap-1.5 rounded-full bg-zinc-100 px-2.5 py-1 text-xs font-medium text-zinc-600">
        <Icon icon={UserGroupIcon} size={14} />
        {count} expert{count === 1 ? "" : "s"}
      </span>
    </button>
  );
}
