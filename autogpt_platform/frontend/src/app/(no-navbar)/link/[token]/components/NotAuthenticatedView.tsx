import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";

interface Props {
  token: string;
  loginRedirect: string;
}

export function NotAuthenticatedView({ token, loginRedirect }: Props) {
  return (
    <AuthCard title="Sign in to continue">
      <div className="flex w-full flex-col items-center gap-6">
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          Sign in to your AutoGPT account to finish setting up AutoPilot.
        </Text>
        <Button as="NextLink" href={loginRedirect} className="w-full">
          Sign in
        </Button>
        <AuthCard.BottomText
          text="Don't have an account?"
          link={{ text: "Sign up", href: `/signup?next=/link/${token}` }}
        />
      </div>
    </AuthCard>
  );
}
