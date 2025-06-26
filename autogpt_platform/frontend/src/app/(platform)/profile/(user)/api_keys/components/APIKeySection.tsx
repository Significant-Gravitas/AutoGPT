"use client";
import { useState } from "react";
import { LuCopy } from "react-icons/lu";
import { Loader2, MoreVertical } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  getGetV1ListUserApiKeysQueryKey,
  useDeleteV1RevokeApiKey,
  useGetV1ListUserApiKeys,
  usePostV1CreateNewApiKey,
} from "@/api/__generated__/endpoints/api-keys/api-keys";
import { APIKeyWithoutHash } from "@/api/__generated__/models/aPIKeyWithoutHash";
import { CreateAPIKeyResponse } from "@/api/__generated__/models/createAPIKeyResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { APIKeyPermission } from "@/api/__generated__/models/aPIKeyPermission";

export function APIKeysSection() {
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isKeyDialogOpen, setIsKeyDialogOpen] = useState(false);
  const [keyState, setKeyState] = useState({
    newKeyName: "",
    newKeyDescription: "",
    newApiKey: "",
    selectedPermissions: [] as APIKeyPermission[],
  });

  const { toast } = useToast();
  const queryClient = getQueryClient();

  const { data: apiKeys, isLoading } = useGetV1ListUserApiKeys({
    query: {
      select: (res) => {
        return (res.data as APIKeyWithoutHash[]).filter(
          (key) => key.status === "ACTIVE",
        );
      },
    },
  });

  const { mutateAsync: createAPIKey } = usePostV1CreateNewApiKey({
    mutation: {
      onSettled: () => {
        return queryClient.invalidateQueries({
          queryKey: getGetV1ListUserApiKeysQueryKey(),
        });
      },
    },
  });

  const { mutateAsync: revokeAPIKey } = useDeleteV1RevokeApiKey({
    mutation: {
      onSettled: () => {
        return queryClient.invalidateQueries({
          queryKey: getGetV1ListUserApiKeysQueryKey(),
        });
      },
    },
  });

  const handleCreateKey = async () => {
    try {
      const response = await createAPIKey({
        data: {
          name: keyState.newKeyName,
          permissions: keyState.selectedPermissions,
          description: keyState.newKeyDescription,
        },
      });
      setKeyState((prev) => ({
        ...prev,
        newApiKey: (response.data as CreateAPIKeyResponse).plain_text_key,
      }));
      setIsCreateOpen(false);
      setIsKeyDialogOpen(true);
    } catch {
      toast({
        title: "Error",
        description: "Failed to create AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  const handleCopyKey = () => {
    navigator.clipboard.writeText(keyState.newApiKey);
    toast({
      title: "Copied",
      description: "AutoGPT Platform API key copied to clipboard",
    });
  };

  const handleRevokeKey = async (keyId: string) => {
    try {
      await revokeAPIKey({
        keyId: keyId,
      });

      toast({
        title: "Success",
        description: "AutoGPT Platform API key revoked successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to revoke AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  return (
    <>
      <div className="mb-4 flex justify-end">
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button>Create Key</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New API Key</DialogTitle>
              <DialogDescription>
                Create a new AutoGPT Platform API key
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  value={keyState.newKeyName}
                  onChange={(e) =>
                    setKeyState((prev) => ({
                      ...prev,
                      newKeyName: e.target.value,
                    }))
                  }
                  placeholder="My AutoGPT Platform API Key"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="description">Description (Optional)</Label>
                <Input
                  id="description"
                  value={keyState.newKeyDescription}
                  onChange={(e) =>
                    setKeyState((prev) => ({
                      ...prev,
                      newKeyDescription: e.target.value,
                    }))
                  }
                  placeholder="Used for..."
                />
              </div>
              <div className="grid gap-2">
                <Label>Permissions</Label>
                {Object.values(APIKeyPermission).map((permission) => (
                  <div className="flex items-center space-x-2" key={permission}>
                    <Checkbox
                      id={permission}
                      checked={keyState.selectedPermissions.includes(
                        permission,
                      )}
                      onCheckedChange={(checked: boolean) => {
                        setKeyState((prev) => ({
                          ...prev,
                          selectedPermissions: checked
                            ? [...prev.selectedPermissions, permission]
                            : prev.selectedPermissions.filter(
                                (p) => p !== permission,
                              ),
                        }));
                      }}
                    />
                    <Label htmlFor={permission}>{permission}</Label>
                  </div>
                ))}
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateKey}>Create</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={isKeyDialogOpen} onOpenChange={setIsKeyDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>AutoGPT Platform API Key Created</DialogTitle>
              <DialogDescription>
                Please copy your AutoGPT API key now. You won&apos;t be able to
                see it again!
              </DialogDescription>
            </DialogHeader>
            <div className="flex items-center space-x-2">
              <code className="flex-1 rounded-md bg-secondary p-2 text-sm">
                {keyState.newApiKey}
              </code>
              <Button size="icon" variant="outline" onClick={handleCopyKey}>
                <LuCopy className="h-4 w-4" />
              </Button>
            </div>
            <DialogFooter>
              <Button onClick={() => setIsKeyDialogOpen(false)}>Close</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading ? (
        <div className="flex justify-center p-4">
          <Loader2 className="h-6 w-6 animate-spin" />
        </div>
      ) : (
        apiKeys &&
        apiKeys.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>API Key</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {apiKeys.map((key) => (
                <TableRow key={key.id}>
                  <TableCell>{key.name}</TableCell>
                  <TableCell>
                    <div className="rounded-md border p-1 px-2 text-xs">
                      {`${key.prefix}******************${key.postfix}`}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        key.status === "ACTIVE" ? "default" : "destructive"
                      }
                      className={
                        key.status === "ACTIVE"
                          ? "border-green-600 bg-green-100 text-green-800"
                          : "border-red-600 bg-red-100 text-red-800"
                      }
                    >
                      {key.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {new Date(key.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    {key.last_used_at
                      ? new Date(key.last_used_at).toLocaleDateString()
                      : "Never"}
                  </TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={() => handleRevokeKey(key.id)}
                        >
                          Revoke
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )
      )}
    </>
  );
}
