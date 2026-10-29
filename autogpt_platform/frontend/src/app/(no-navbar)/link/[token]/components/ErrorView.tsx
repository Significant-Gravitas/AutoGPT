import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { Unlink01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  message: string;
}

export function ErrorView({ message }: Props) {
  return (
    <AuthCard title="Setup failed">
      <div className="flex w-full flex-col items-center gap-6">
        <div
          // TODO(tokens): no exact role for bg-red-100 (#FDECEC)
          className="flex h-16 w-16 items-center justify-center rounded-full bg-red-100"
        >
          <Icon
            icon={Unlink01Icon}
            size={40}
            // TODO(tokens): no fitting role for text-red-600 #D93636; #D93636 is error-hover (a hover fill), not a text colour
            className="text-red-600"
          />
        </div>
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          {message}
        </Text>
        <Text variant="small" className="text-center text-muted-foreground">
          Go back to your chat and ask the bot for a new setup link.
        </Text>
      </div>
    </AuthCard>
  );
}
