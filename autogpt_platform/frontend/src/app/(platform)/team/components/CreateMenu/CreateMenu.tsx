"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { MARKETPLACE_EXPERTS_HREF } from "@/lib/constants";
import {
  FlowIcon,
  PlusSignIcon,
  SparklesIcon,
  UserAdd01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

interface Props {
  onNewPod: () => void;
}

export function CreateMenu({ onNewPod }: Props) {
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
          <Link href={MARKETPLACE_EXPERTS_HREF}>
            <Icon icon={UserAdd01Icon} className="mr-2 size-4" />
            Hire an expert
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/raise">
            <Icon icon={SparklesIcon} className="mr-2 size-4" />
            Raise your own expert
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href="/build">
            <Icon icon={FlowIcon} className="mr-2 size-4" />
            Build an agent from scratch
          </Link>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onNewPod}>
          <Icon icon={UserGroupIcon} className="mr-2 size-4" />
          New pod
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
