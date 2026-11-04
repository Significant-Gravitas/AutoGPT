import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  description: string;
}

/** What the person asked for, in their words, right under the title. */
export function OriginalAsk({ title, description }: Props) {
  const ask = description.trim();
  if (!ask || ask === title.trim()) return null;

  return (
    <section aria-label="What was asked" className="flex flex-col gap-1">
      <Text as="h3" variant="small" tone="faint">
        Asked
      </Text>
      <Text
        variant="inbox"
        tone="text"
        className="text-pretty break-words whitespace-pre-wrap"
      >
        {ask}
      </Text>
    </section>
  );
}
