import { Card as KobraCard } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Props = React.ComponentProps<typeof KobraCard>;

// The house Card is a padded surface whose children flow normally, so Kobra's
// flex column (meant for CardHeader/CardContent/CardFooter) is switched off.
export function Card({ className, ...props }: Props) {
  return <KobraCard className={cn("block p-6", className)} {...props} />;
}
