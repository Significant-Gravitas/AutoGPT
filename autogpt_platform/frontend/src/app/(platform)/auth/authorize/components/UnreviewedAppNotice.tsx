import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";

interface Props {
  redirectURI: string;
}

export function UnreviewedAppNotice({ redirectURI }: Props) {
  return (
    <Alert variant="warning">
      <AlertTitle>AutoGPT hasn&apos;t reviewed this app</AlertTitle>
      <AlertDescription>
        It added itself to ask for access, so its name could be anything. Go
        ahead only if you just started this yourself, from an app on this
        computer. Afterwards you&apos;ll be sent back to it at{" "}
        <strong>{hostOf(redirectURI)}</strong>.
      </AlertDescription>
    </Alert>
  );
}

function hostOf(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
