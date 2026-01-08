"use client";

import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import { Button } from "@/components/__legacy__/ui/button";
import { Input } from "@/components/__legacy__/ui/input";
import { Badge } from "@/components/__legacy__/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/__legacy__/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/__legacy__/ui/dropdown-menu";
import { PaginationControls } from "@/components/__legacy__/ui/pagination-controls";
import { MoreHorizontal, Copy, Eye, EyeOff, Search, Plus, RefreshCw, Trash2, Edit } from "lucide-react";
import type { OAuthApplication, OAuthAppsListResponse } from "@/lib/autogpt-server-api/types";
import { deleteOAuthApp, regenerateOAuthSecret, updateOAuthApp } from "../actions";
import { CreateOAuthAppDialog } from "./CreateOAuthAppDialog";
import { useRouter, useSearchParams } from "next/navigation";

interface OAuthAppListProps {
  initialData: OAuthAppsListResponse;
  initialSearch?: string;
}

export function OAuthAppList({ initialData, initialSearch }: OAuthAppListProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [searchQuery, setSearchQuery] = useState(initialSearch || "");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedApp, setSelectedApp] = useState<OAuthApplication | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isRegenerateDialogOpen, setIsRegenerateDialogOpen] = useState(false);
  const [newSecret, setNewSecret] = useState<string | null>(null);
  const [isSecretVisible, setIsSecretVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = () => {
    const params = new URLSearchParams(searchParams.toString());
    if (searchQuery) {
      params.set("search", searchQuery);
    } else {
      params.delete("search");
    }
    params.set("page", "1");
    router.push(`/admin/oauth?${params.toString()}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const handleDelete = async () => {
    if (!selectedApp) return;
    setIsLoading(true);
    try {
      await deleteOAuthApp(selectedApp.id);
      setIsDeleteDialogOpen(false);
      setSelectedApp(null);
    } catch (error) {
      console.error("Failed to delete OAuth app:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegenerateSecret = async () => {
    if (!selectedApp) return;
    setIsLoading(true);
    try {
      const result = await regenerateOAuthSecret(selectedApp.id);
      setNewSecret(result.client_secret);
      setIsSecretVisible(true);
    } catch (error) {
      console.error("Failed to regenerate secret:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleActive = async (app: OAuthApplication) => {
    setIsLoading(true);
    try {
      await updateOAuthApp(app.id, { is_active: !app.is_active });
    } catch (error) {
      console.error("Failed to toggle app status:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatDate = (dateStr: string) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(new Date(dateStr));
  };

  return (
    <div className="space-y-4">
      {/* Search and Create */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-1 items-center gap-2">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search by name, client ID, or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="pl-10"
            />
          </div>
          <Button onClick={handleSearch} variant="secondary">
            Search
          </Button>
        </div>
        <Button onClick={() => setIsCreateDialogOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create OAuth App
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-md border bg-white">
        <Table>
          <TableHeader className="bg-gray-50">
            <TableRow>
              <TableHead className="font-medium">Name</TableHead>
              <TableHead className="font-medium">Client ID</TableHead>
              <TableHead className="font-medium">Scopes</TableHead>
              <TableHead className="font-medium">Status</TableHead>
              <TableHead className="font-medium">Created</TableHead>
              <TableHead className="text-right font-medium">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {initialData.applications.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="py-10 text-center text-gray-500">
                  No OAuth applications found
                </TableCell>
              </TableRow>
            ) : (
              initialData.applications.map((app) => (
                <TableRow key={app.id} className="hover:bg-gray-50">
                  <TableCell>
                    <div>
                      <div className="font-medium">{app.name}</div>
                      {app.description && (
                        <div className="text-sm text-gray-500 truncate max-w-xs">
                          {app.description}
                        </div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                        {app.client_id.slice(0, 20)}...
                      </code>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => copyToClipboard(app.client_id)}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1 max-w-xs">
                      {app.scopes.slice(0, 2).map((scope) => (
                        <Badge key={scope} variant="secondary" className="text-xs">
                          {scope}
                        </Badge>
                      ))}
                      {app.scopes.length > 2 && (
                        <Badge variant="secondary" className="text-xs">
                          +{app.scopes.length - 2}
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={app.is_active ? "default" : "secondary"}
                      className={app.is_active ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"}
                    >
                      {app.is_active ? "Active" : "Inactive"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-gray-600">
                    {formatDate(app.created_at)}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="sm">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleToggleActive(app)}>
                          {app.is_active ? (
                            <>
                              <EyeOff className="mr-2 h-4 w-4" />
                              Disable
                            </>
                          ) : (
                            <>
                              <Eye className="mr-2 h-4 w-4" />
                              Enable
                            </>
                          )}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => {
                            setSelectedApp(app);
                            setIsRegenerateDialogOpen(true);
                          }}
                        >
                          <RefreshCw className="mr-2 h-4 w-4" />
                          Regenerate Secret
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={() => {
                            setSelectedApp(app);
                            setIsDeleteDialogOpen(true);
                          }}
                          className="text-red-600"
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <PaginationControls
        currentPage={initialData.page}
        totalPages={initialData.total_pages}
      />

      {/* Create Dialog */}
      <CreateOAuthAppDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
      />

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete OAuth Application</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete <strong>{selectedApp?.name}</strong>?
              This will also delete all associated authorization codes, access tokens,
              and refresh tokens. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-red-600 hover:bg-red-700"
              disabled={isLoading}
            >
              {isLoading ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Regenerate Secret Dialog */}
      <Dialog
        open={isRegenerateDialogOpen}
        onOpenChange={(open) => {
          setIsRegenerateDialogOpen(open);
          if (!open) {
            setNewSecret(null);
            setIsSecretVisible(false);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {newSecret ? "New Client Secret Generated" : "Regenerate Client Secret"}
            </DialogTitle>
            <DialogDescription>
              {newSecret
                ? "Your new client secret is shown below. Make sure to copy it now - you won't be able to see it again!"
                : `Are you sure you want to regenerate the client secret for ${selectedApp?.name}? The old secret will be invalidated immediately.`}
            </DialogDescription>
          </DialogHeader>

          {newSecret && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
                <div className="flex-1">
                  <label className="text-sm font-medium text-gray-700">Client Secret</label>
                  <div className="flex items-center gap-2 mt-1">
                    <code className="flex-1 text-sm bg-gray-100 px-3 py-2 rounded break-all">
                      {isSecretVisible ? newSecret : "••••••••••••••••••••••••••••••••"}
                    </code>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsSecretVisible(!isSecretVisible)}
                    >
                      {isSecretVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(newSecret)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
              <p className="text-sm text-yellow-700 bg-yellow-50 p-3 rounded">
                This secret will only be shown once. Store it securely!
              </p>
            </div>
          )}

          <DialogFooter>
            {newSecret ? (
              <Button onClick={() => setIsRegenerateDialogOpen(false)}>Done</Button>
            ) : (
              <>
                <Button variant="secondary" onClick={() => setIsRegenerateDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleRegenerateSecret} disabled={isLoading}>
                  {isLoading ? "Regenerating..." : "Regenerate Secret"}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
