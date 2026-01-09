"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { updateWaitlist } from "../actions";
import type {
  WaitlistAdminResponse,
  WaitlistUpdateRequest,
} from "@/lib/autogpt-server-api/types";
import { useToast } from "@/components/molecules/Toast/use-toast";

type EditWaitlistDialogProps = {
  waitlist: WaitlistAdminResponse;
  onClose: () => void;
  onSave: () => void;
};

export function EditWaitlistDialog({
  waitlist,
  onClose,
  onSave,
}: EditWaitlistDialogProps) {
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const [formData, setFormData] = useState({
    name: waitlist.name,
    slug: waitlist.slug,
    subHeading: waitlist.subHeading,
    description: waitlist.description,
    categories: waitlist.categories.join(", "),
    imageUrls: waitlist.imageUrls.join(", "),
    videoUrl: waitlist.videoUrl || "",
    agentOutputDemoUrl: waitlist.agentOutputDemoUrl || "",
    status: waitlist.status,
    storeListingId: waitlist.storeListingId || "",
  });

  function handleChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  }

  function handleStatusChange(value: string) {
    setFormData((prev) => ({
      ...prev,
      status: value,
    }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);

    try {
      const updateData: WaitlistUpdateRequest = {
        name: formData.name,
        slug: formData.slug,
        subHeading: formData.subHeading,
        description: formData.description,
        categories: formData.categories
          ? formData.categories.split(",").map((c) => c.trim())
          : [],
        imageUrls: formData.imageUrls
          ? formData.imageUrls.split(",").map((u) => u.trim())
          : [],
        videoUrl: formData.videoUrl || null,
        agentOutputDemoUrl: formData.agentOutputDemoUrl || null,
        status: formData.status,
        storeListingId: formData.storeListingId || null,
      };

      await updateWaitlist(waitlist.id, updateData);

      toast({
        title: "Success",
        description: "Waitlist updated successfully",
      });

      onSave();
    } catch (error) {
      console.error("Error updating waitlist:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update waitlist",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Edit Waitlist</DialogTitle>
          <DialogDescription>
            Update the waitlist details. Changes will be reflected immediately.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="slug">Slug</Label>
              <Input
                id="slug"
                name="slug"
                value={formData.slug}
                onChange={handleChange}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="subHeading">Subheading</Label>
              <Input
                id="subHeading"
                name="subHeading"
                value={formData.subHeading}
                onChange={handleChange}
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleChange}
                rows={4}
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="status">Status</Label>
              <Select
                value={formData.status}
                onValueChange={handleStatusChange}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="NOT_STARTED">Not Started</SelectItem>
                  <SelectItem value="WORK_IN_PROGRESS">
                    Work In Progress
                  </SelectItem>
                  <SelectItem value="DONE">Done</SelectItem>
                  <SelectItem value="CANCELED">Canceled</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="categories">Categories (comma-separated)</Label>
              <Input
                id="categories"
                name="categories"
                value={formData.categories}
                onChange={handleChange}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="imageUrls">Image URLs (comma-separated)</Label>
              <Input
                id="imageUrls"
                name="imageUrls"
                value={formData.imageUrls}
                onChange={handleChange}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="videoUrl">Video URL</Label>
              <Input
                id="videoUrl"
                name="videoUrl"
                value={formData.videoUrl}
                onChange={handleChange}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="agentOutputDemoUrl">Output Demo URL</Label>
              <Input
                id="agentOutputDemoUrl"
                name="agentOutputDemoUrl"
                value={formData.agentOutputDemoUrl}
                onChange={handleChange}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="storeListingId">
                Store Listing ID (for linking)
              </Label>
              <Input
                id="storeListingId"
                name="storeListingId"
                value={formData.storeListingId}
                onChange={handleChange}
                placeholder="Leave empty if not linked"
              />
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" loading={loading}>
              Save Changes
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
