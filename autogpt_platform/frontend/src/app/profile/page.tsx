"use client";

import { useSupabase } from "@/components/SupabaseProvider";
import { Button } from "@/components/ui/button";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { useCallback, useContext, useMemo, useState } from "react";
import { FaSpinner } from "react-icons/fa";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/components/ui/use-toast";
import { IconKey, IconUser } from "@/components/ui/icons";
import { LogOutIcon, Trash2Icon } from "lucide-react";
import { providerIcons } from "@/components/integrations/credentials-input";
import { CredentialsProvidersContext } from "@/components/integrations/credentials-provider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CredentialsProviderName } from "@/lib/autogpt-server-api";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

export default function PrivatePage() {
  const { user, isLoading, error } = useUser();
  const { supabase } = useSupabase();
  const router = useRouter();
  const providers = useContext(CredentialsProvidersContext);
  const { toast } = useToast();

  const [confirmationDialogState, setConfirmationDialogState] = useState<
    | {
        open: true;
        message: string;
        onConfirm: () => void;
        onReject: () => void;
      }
    | { open: false }
  >({ open: false });

  const removeCredentials = useCallback(
    async (
      provider: CredentialsProviderName,
      id: string,
      force: boolean = false,
    ) => {
      if (!providers || !providers[provider]) {
        return;
      }

      let result;
      try {
        result = await providers[provider].deleteCredentials(id, force);
      } catch (error: any) {
        toast({
          title: "Something went wrong when deleting credentials: " + error,
          variant: "destructive",
          duration: 2000,
        });
        setConfirmationDialogState({ open: false });
        return;
      }
      if (result.deleted) {
        if (result.revoked) {
          toast({
            title: "Credentials deleted",
            duration: 2000,
          });
        } else {
          toast({
            title: "Credentials deleted from AutoGPT",
            description: `You may also manually remove the connection to AutoGPT at ${provider}!`,
            duration: 3000,
          });
        }
        setConfirmationDialogState({ open: false });
      } else if (result.need_confirmation) {
        setConfirmationDialogState({
          open: true,
          message: result.message,
          onConfirm: () => removeCredentials(provider, id, true),
          onReject: () => setConfirmationDialogState({ open: false }),
        });
      }
    },
    [providers, toast],
  );

  //TODO: remove when the way system credentials are handled is updated
  // This contains ids for built-in "Use Credits for X" credentials
  const hiddenCredentials = useMemo(
    () => [
      "fdb7f412-f519-48d1-9b5f-d2f73d0e01fe", // Revid
      "760f84fc-b270-42de-91f6-08efe1b512d0", // Ideogram
      "6b9fc200-4726-4973-86c9-cd526f5ce5db", // Replicate
      "53c25cb8-e3ee-465c-a4d1-e75a4c899c2a", // OpenAI
      "24e5d942-d9e3-4798-8151-90143ee55629", // Anthropic
      "4ec22295-8f97-4dd1-b42b-2c6957a02545", // Groq
      "7f7b0654-c36b-4565-8fa7-9a52575dfae2", // D-ID
      "7f26de70-ba0d-494e-ba76-238e65e7b45f", // Jina
      "66f20754-1b81-48e4-91d0-f4f0dd82145f", // Unreal Speech
      "b5a0e27d-0c98-4df3-a4b9-10193e1f3c40", // Open Router
    ],
    [],
  );

  if (isLoading || !providers) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
      </div>
    );
  }

  if (error || !user || !supabase) {
    router.push("/login");
    return null;
  }

  const allCredentials = Object.values(providers).flatMap((provider) =>
    [...provider.savedOAuthCredentials, ...provider.savedApiKeys]
      .filter((cred) => !hiddenCredentials.includes(cred.id))
      .map((credentials) => ({
        ...credentials,
        provider: provider.provider,
        providerName: provider.providerName,
        ProviderIcon: providerIcons[provider.provider],
        TypeIcon: { oauth2: IconUser, api_key: IconKey }[credentials.type],
      })),
  );

  return (
    <div className="mx-auto max-w-3xl md:py-8">
      <div className="flex items-center justify-between">
        <p>
          Hello <span data-testid="profile-email">{user.email}</span>
        </p>
        <Button onClick={() => supabase.auth.signOut()}>
          <LogOutIcon className="mr-1.5 size-4" />
          Log out
        </Button>
      </div>
      <Separator className="my-6" />
      <h2 className="mb-4 text-lg">Connections & Credentials</h2>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Provider</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {allCredentials.map((cred) => (
            <TableRow key={cred.id}>
              <TableCell>
                <div className="flex items-center space-x-1.5">
                  <cred.ProviderIcon className="h-4 w-4" />
                  <strong>{cred.providerName}</strong>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex h-full items-center space-x-1.5">
                  <cred.TypeIcon />
                  <span>{cred.title || cred.username}</span>
                </div>
                <small className="text-muted-foreground">
                  {
                    {
                      oauth2: "OAuth2 credentials",
                      api_key: "API key",
                    }[cred.type]
                  }{" "}
                  - <code>{cred.id}</code>
                </small>
              </TableCell>
              <TableCell className="w-0 whitespace-nowrap">
                <Button
                  variant="destructive"
                  onClick={() => removeCredentials(cred.provider, cred.id)}
                >
                  <Trash2Icon className="mr-1.5 size-4" /> Delete
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <AlertDialog open={confirmationDialogState.open}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              {confirmationDialogState.open && confirmationDialogState.message}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel
              onClick={() =>
                confirmationDialogState.open &&
                confirmationDialogState.onReject()
              }
            >
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() =>
                confirmationDialogState.open &&
                confirmationDialogState.onConfirm()
              }
            >
              Continue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
