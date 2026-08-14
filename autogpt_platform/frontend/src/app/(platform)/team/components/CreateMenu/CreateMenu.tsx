"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  FlowIcon,
  PlusSignIcon,
  UserAdd01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

export function CreateMenu() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          size="small"
          leftIcon={<Icon icon={PlusSignIcon} className="size-4" />}
        >
          Create
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuItem asChild>
          <Link href="/marketplace#experts">
            <Icon icon={UserAdd01Icon} className="mr-2 size-4" />
            Hire an expert
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/build">
            <Icon icon={FlowIcon} className="mr-2 size-4" />
            Build an agent from scratch
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
