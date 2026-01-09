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
  DialogTrigger,
} from "@/components/__legacy__/ui/dialog";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { Textarea } from "@/components/__legacy__/ui/textarea";
import { createWaitlist } from "../actions";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { Plus } from "lucide-react";

export function CreateWaitlistButton() {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const router = useRouter();

  const [formData, setFormData] = useState({
    name: "",
    slug: "",
    subHeading: "",
    description: "",
    categories: "",
    imageUrls: "",
    videoUrl: "",
    agentOutputDemoUrl: "",
  });

  function handleChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  }

  function generateSlug(name: string) {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);

    try {
      await createWaitlist({
        name: formData.name,
        slug: formData.slug || generateSlug(formData.name),
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
      });

      toast({
        title: "Success",
        description: "Waitlist created successfully",
      });

      setOpen(false);
      setFormData({
        name: "",
        slug: "",
        subHeading: "",
        description: "",
        categories: "",
        imageUrls: "",
        videoUrl: "",
        agentOutputDemoUrl: "",
      });
      router.refresh();
    } catch (error) {
      console.error("Error creating waitlist:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to create waitlist",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Create Waitlist
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Create New Waitlist</DialogTitle>
          <DialogDescription>
            Create a new waitlist for an upcoming agent. Users can sign up to be
            notified when it launches.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name *</Label>
              <Input
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="SEO Analysis Agent"
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
                placeholder="seo-analysis-agent (auto-generated if empty)"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="subHeading">Subheading *</Label>
              <Input
                id="subHeading"
                name="subHeading"
                value={formData.subHeading}
                onChange={handleChange}
                placeholder="Analyze your website's SEO in minutes"
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="description">Description *</Label>
              <Textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleChange}
                placeholder="Detailed description of what this agent does..."
                rows={4}
                required
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="categories">Categories (comma-separated)</Label>
              <Input
                id="categories"
                name="categories"
                value={formData.categories}
                onChange={handleChange}
                placeholder="SEO, Marketing, Analysis"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="imageUrls">Image URLs (comma-separated)</Label>
              <Input
                id="imageUrls"
                name="imageUrls"
                value={formData.imageUrls}
                onChange={handleChange}
                placeholder="https://example.com/image1.jpg, https://example.com/image2.jpg"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="videoUrl">Video URL (optional)</Label>
              <Input
                id="videoUrl"
                name="videoUrl"
                value={formData.videoUrl}
                onChange={handleChange}
                placeholder="https://youtube.com/watch?v=..."
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="agentOutputDemoUrl">
                Output Demo URL (optional)
              </Label>
              <Input
                id="agentOutputDemoUrl"
                name="agentOutputDemoUrl"
                value={formData.agentOutputDemoUrl}
                onChange={handleChange}
                placeholder="https://example.com/demo-output.mp4"
              />
            </div>
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="secondary"
              onClick={() => setOpen(false)}
            >
              Cancel
            </Button>
            <Button type="submit" loading={loading}>
              Create Waitlist
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
