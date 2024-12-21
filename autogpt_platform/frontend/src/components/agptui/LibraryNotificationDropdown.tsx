"use client";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "./Button";
import { BellIcon, X } from "lucide-react";
import { motion, useAnimationControls } from "framer-motion";
import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../ui/card";

export const LibraryNotificationDropdown = () => {
  const controls = useAnimationControls();
  const [open, setOpen] = useState(false);

  const handleHoverStart = () => {
    controls.start({
      rotate: [0, -10, 10, -10, 10, 0],
      transition: { duration: 0.5 },
    });
  };
  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant={open ? "library_primary" : "library_outline"}
          size="library"
          onMouseEnter={handleHoverStart}
          onMouseLeave={handleHoverStart}
          className="w-[161px]"
        >
          <motion.div animate={controls}>
            <BellIcon className="mr-2 h-5 w-5" strokeWidth={2} />
          </motion.div>
          Your updates
          <span className="ml-2 text-[14px]">2</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="scroll-none relative left-[16px] h-[80vh] w-fit overflow-y-auto rounded-[26px] bg-[#a1a1aa]/60 p-5">
        <DropdownMenuLabel className="mb-4 font-sans text-[18px] text-white">
          Agent run updates
        </DropdownMenuLabel>
        <button
          className="absolute right-[10px] top-[20px] h-fit w-fit"
          onClick={() => setOpen(false)}
        >
          <X className="h-6 w-6 text-white hover:text-white/60" />
        </button>
        <DropdownMenuItem>
          <LibraryNotificationCard />
        </DropdownMenuItem>
        <DropdownMenuItem>
          <LibraryNotificationCard />
        </DropdownMenuItem>
        <DropdownMenuItem>
          <LibraryNotificationCard />
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

const LibraryNotificationCard = () => {
  return (
    <Card className="w-[424px] rounded-[14px] border border-neutral-100 p-[16px] pt-[12px]">
      <CardHeader>
        <CardTitle>Latest Agent Updates</CardTitle>
        <CardDescription>View your latest workflow changes</CardDescription>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-full bg-neutral-100" />
            <div>
              <p className="font-medium">Agent Run #1234</p>
              <p className="text-sm text-neutral-500">Updated 2 hours ago</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-full bg-neutral-100" />
            <div>
              <p className="font-medium">Workflow Changes</p>
              <p className="text-sm text-neutral-500">3 new changes detected</p>
            </div>
          </div>
        </div>
      </CardContent>

      <CardFooter>
        <Button variant="outline" className="w-full">
          View All Updates
        </Button>
      </CardFooter>
    </Card>
  );
};
