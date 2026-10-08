import { Skeleton as KobraSkeleton } from "@/components/ui/skeleton";

type Props = React.ComponentProps<typeof KobraSkeleton>;

export function Skeleton(props: Props) {
  return <KobraSkeleton {...props} />;
}
