import { Text } from "@/components/atoms/Text/Text";

interface Props {
  label: string;
  value: string | null;
}

/** One labelled paragraph on a Basics tab, shared by the expert and
 *  Autopilot pages so their headings stay the same size. */
export function ProfileEntry({ label, value }: Props) {
  return (
    <div>
      <Text variant="large-medium" as="dt" tone="primary">
        {label}
      </Text>
      <Text
        variant="body"
        as="dd"
        tone="secondary"
        className="mt-1 whitespace-pre-line"
      >
        {value || "Not set yet."}
      </Text>
    </div>
  );
}
