"use client";

import { useState, useEffect } from "react";

import { APIKey, APIKeyPermission } from "@/lib/autogpt-server-api/types";

import { LuCopy } from "react-icons/lu";
import { Loader2, MoreVertical } from "lucide-react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/ui/use-toast";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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

export function APIKeysSection() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isKeyDialogOpen, setIsKeyDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyDescription, setNewKeyDescription] = useState("");
  const [newApiKey, setNewApiKey] = useState("");
  const [selectedPermissions, setSelectedPermissions] = useState<
    APIKeyPermission[]
  >([]);
  const { toast } = useToast();
  const api = useBackendAPI();

  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    setIsLoading(true);
    try {
      const keys = await api.listAPIKeys();
      setApiKeys(keys.filter((key) => key.status === "ACTIVE"));
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateKey = async () => {
    try {
      const response = await api.createAPIKey(
        newKeyName,
        selectedPermissions,
        newKeyDescription,
      );

      setNewApiKey(response.plain_text_key);
      setIsCreateOpen(false);
      setIsKeyDialogOpen(true);
      loadAPIKeys();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  const handleCopyKey = () => {
    navigator.clipboard.writeText(newApiKey);
    toast({
      title: "Copied",
      description: "AutoGPT Platform API key copied to clipboard",
    });
  };

  const handleRevokeKey = async (keyId: string) => {
    try {
      await api.revokeAPIKey(keyId);
      toast({
        title: "Success",
        description: "AutoGPT Platform API key revoked successfully",
      });
      loadAPIKeys();
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to revoke AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>AutoGPT Platform API Keys</CardTitle>
        <CardDescription>
          Manage your AutoGPT Platform API keys for programmatic access
        </CardDescription>
      </CardHeader>
      <CardContent>
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
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    placeholder="My AutoGPT Platform API Key"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="description">Description (Optional)</Label>
                  <Input
                    id="description"
                    value={newKeyDescription}
                    onChange={(e) => setNewKeyDescription(e.target.value)}
                    placeholder="Used for..."
                  />
                </div>
                <div className="grid gap-2">
                  <Label>Permissions</Label>
                  {Object.values(APIKeyPermission).map((permission) => (
                    <div
                      className="flex items-center space-x-2"
                      key={permission}
                    >
                      <Checkbox
                        id={permission}
                        checked={selectedPermissions.includes(permission)}
                        onCheckedChange={(checked) => {
                          setSelectedPermissions(
                            checked
                              ? [...selectedPermissions, permission]
                              : selectedPermissions.filter(
                                  (p) => p !== permission,
                                ),
                          );
                        }}
                      />
                      <Label htmlFor={permission}>{permission}</Label>
                    </div>
                  ))}
                </div>
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setIsCreateOpen(false)}
                >
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
                  Please copy your AutoGPT API key now. You won&apos;t be able
                  to see it again!
                </DialogDescription>
              </DialogHeader>
              <div className="flex items-center space-x-2">
                <code className="flex-1 rounded-md bg-secondary p-2 text-sm">
                  {newApiKey}
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
      </CardContent>
    </Card>
  );
}
