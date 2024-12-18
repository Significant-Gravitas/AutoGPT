"use client";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "./ui/button";
import { useRouter } from "next/navigation";
import useSupabase from "@/hooks/useSupabase";

const ProfileDropdown = () => {
  const { supabase, user, isUserLoading } = useSupabase();
  const router = useRouter();

  if (isUserLoading) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="h-8 w-8 rounded-full">
          <Avatar>
            <AvatarImage
              src={user?.user_metadata["avatar_url"]}
              alt="User Avatar"
            />
            <AvatarFallback>CN</AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => router.push("/profile")}>
          Profile
        </DropdownMenuItem>
        {user!.role === "admin" && (
          <DropdownMenuItem onClick={() => router.push("/admin/dashboard")}>
            Admin Dashboard
          </DropdownMenuItem>
        )}
        <DropdownMenuItem
          onClick={() =>
            supabase?.auth.signOut().then(() => router.replace("/login"))
          }
        >
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default ProfileDropdown;
