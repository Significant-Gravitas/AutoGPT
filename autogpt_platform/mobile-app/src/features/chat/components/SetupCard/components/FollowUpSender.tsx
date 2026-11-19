import { useMountEffect } from "@/lib/useMountEffect";

interface Props {
  onReady: () => void;
}

/** Mounted only once the card's reply may go out; sends it once on mount. */
export function FollowUpSender({ onReady }: Props) {
  useMountEffect(() => {
    onReady();
  });
  return null;
}
