import { Text } from "@/components/atoms/Text/Text";

interface Props {
  usePaused?: boolean;
}

export function SourceRemovedNotice({ usePaused }: Props) {
  return (
    <Text variant="small" tone="danger" data-testid="source-removed-notice">
      A source this version was learned from is no longer available, so Otto
      stopped using this skill. To use it again, edit the skill, or restore an
      earlier version from History if there is one
      {usePaused ? ", then choose Resume using this skill" : ""}.
    </Text>
  );
}
