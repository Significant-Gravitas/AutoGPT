import { Text } from "@/components/atoms/Text/Text";

interface Props {
  id: string;
  title: string;
  description: string;
  helper?: string;
  children: React.ReactNode;
}

export function SettingRow({
  id,
  title,
  description,
  helper,
  children,
}: Props) {
  return (
    <div className="flex flex-col gap-3 border-b border-zinc-200 py-5 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <Text variant="body-medium" tone="primary" id={`${id}-title`}>
          {title}
        </Text>
        <Text variant="body" tone="secondary" id={`${id}-description`}>
          {description}
        </Text>
        {helper ? (
          <Text variant="small" tone="muted">
            {helper}
          </Text>
        ) : null}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}
