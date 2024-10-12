"use client";

import { useSupabase } from "@/components/SupabaseProvider";
import { Button } from "@/components/ui/button";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { useCallback, useContext, useMemo } from "react";
import { FaSpinner } from "react-icons/fa";
import { CredentialsProvidersContext } from "@/components/integrations/credentials-provider";
import { Separator } from "@/components/ui/separator";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useToast } from "@/components/ui/use-toast";
import { IconKey, IconUser } from "@/components/ui/icons";
import { LogOutIcon, Trash2Icon } from "lucide-react";
import { providerIcons } from "@/components/integrations/credentials-input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function PrivatePage() {
  const { user, isLoading, error } = useUser();
  const { supabase } = useSupabase();
  const router = useRouter();
  const providers = useContext(CredentialsProvidersContext);
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const { toast } = useToast();

  const removeCredentials = useCallback(
    async (provider: string, id: string) => {
      try {
        const response = await api.deleteCredentials(provider, id);
        console.log("response", response);
        toast({
          title: "Credentials deleted",
          duration: 2000,
        });
      } catch (error: any) {
        if (error.response && error.response.status === 501) {
          toast({
            title: "Credentials deleted from AutoGPT",
            description: `You may also manually remove the connection to AutoGPT at ${provider}!`,
            duration: 3000,
          });
          return;
        }

        toast({
          title: "Something went wrong when deleting credentials " + error,
          variant: "destructive",
          duration: 2000,
        });
      }
    },
    [api],
  );

  if (isLoading || !providers || !providers) {
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
    [...provider.savedOAuthCredentials, ...provider.savedApiKeys].map(
      (credentials) => ({
        ...credentials,
        provider: provider.provider,
        providerName: provider.providerName,
        ProviderIcon: providerIcons[provider.provider],
        TypeIcon: { oauth2: IconUser, api_key: IconKey }[credentials.type],
      }),
    ),
  );

  return (
    <div className="max-w-3xl mx-auto md:py-8">
      <div className="flex items-center justify-between">
        <p>Hello {user.email}</p>
        <Button onClick={() => supabase.auth.signOut()}>
          <LogOutIcon className="size-4 mr-1.5" />
          Log out
        </Button>
      </div>
      <Separator className="my-6" />
      <h2 className="text-lg mb-4">Connections & Credentials</h2>
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
                  <Trash2Icon className="size-4 mr-1.5" /> Delete
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
