import { Button } from "../atoms/Button/Button";
import { Text } from "../atoms/Text/Text";

interface Props {
  onBackToLogin?: () => void;
}

export function WaitlistErrorContent(props: Props) {
  return (
    <div className="flex flex-col items-center gap-6">
      <Text variant="h3">We&apos;re in closed beta</Text>
      <div className="flex flex-col gap-4 text-center">
        <Text variant="large" className="text-center">
          Looks like your email isn&apos;t in our early access list just yet.
          Join the waitlist and we will let you know the moment we open up
          access!
        </Text>
      </div>
      <div className="flex gap-2">
        <Button
          onClick={() => {
            window.open("https://agpt.co/waitlist", "_blank");
          }}
        >
          Join Waitlist
        </Button>
        {props.onBackToLogin ? (
          <Button variant="secondary" onClick={props.onBackToLogin}>
            Back to Login
          </Button>
        ) : null}
      </div>
      <div className="flex flex-col gap-2">
        <Text variant="small" className="text-center text-muted-foreground">
          Already joined? Double-check you are using the same email you signed
          up with. Need a hand? Emails us at{" "}
          <a
            href="mailto:contact@agpt.co"
            className="underline hover:text-foreground"
          >
            contact@agpt.co
          </a>{" "}
          or{" "}
          <a
            href="https://discord.gg/autogpt"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-foreground"
          >
            message us on Discord
          </a>
        </Text>
      </div>
    </div>
  );
}
