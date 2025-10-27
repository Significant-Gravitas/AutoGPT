import { Button } from "../atoms/Button/Button";
import { Text } from "../atoms/Text/Text";

interface WaitlistErrorContentProps {
  onClose: () => void;
  closeButtonText?: string;
  closeButtonVariant?: "primary" | "secondary";
}

export function WaitlistErrorContent({
  onClose,
  closeButtonText = "Close",
  closeButtonVariant = "primary",
}: WaitlistErrorContentProps) {
  return (
    <div className="flex flex-col items-center gap-6">
      <Text variant="h3">Join the Waitlist</Text>
      <div className="flex flex-col gap-4 text-center">
        <Text variant="large-medium" className="text-center">
          The AutoGPT Platform is currently in closed beta. Your email address
          isn&apos;t on our current allowlist for early access.
        </Text>
        <Text variant="body" className="text-center">
          Join our waitlist to get notified when we open up access!
        </Text>
      </div>
      <div className="flex gap-3">
        <Button
          variant="secondary"
          onClick={() => {
            window.open("https://agpt.co/waitlist", "_blank");
          }}
        >
          Join Waitlist
        </Button>
        <Button variant={closeButtonVariant} onClick={onClose}>
          {closeButtonText}
        </Button>
      </div>
      <div className="flex flex-col gap-2">
        <Text variant="small" className="text-center text-muted-foreground">
          Already signed up for the waitlist? Make sure you&apos;re using the
          exact same email address you used when signing up.
        </Text>
        <Text variant="small" className="text-center text-muted-foreground">
          If you&apos;re not sure which email you used or need help, contact us
          at{" "}
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
            reach out on Discord
          </a>
        </Text>
      </div>
    </div>
  );
}
