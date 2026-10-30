import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { FEEDBACK_FORM_URL } from "@/components/molecules/TallyPoup/helpers";

interface Props {
  variant?: "kept" | "too_late";
}

export function StillClosedHelp({ variant = "kept" }: Props) {
  return (
    <Text variant="small" tone="muted">
      {variant === "kept"
        ? "Still shown as closed? Wait a minute and sign in again. Still closed after a few minutes? "
        : "Think this is a mistake? "}
      <Link href={FEEDBACK_FORM_URL} isExternal variant="secondary">
        Contact us
      </Link>
      .
    </Text>
  );
}
