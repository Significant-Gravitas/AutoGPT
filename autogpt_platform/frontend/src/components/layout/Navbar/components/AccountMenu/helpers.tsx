import { IconType } from "@/components/__legacy__/ui/icons";
import {
  ChartLineUpIcon,
  CreditCardIcon,
  GearIcon,
  QuestionIcon,
  SignOutIcon,
  SlidersHorizontalIcon,
  UploadSimpleIcon,
  UserIcon,
} from "@phosphor-icons/react";

export function getAccountMenuPhosphorIcon(icon: IconType) {
  const className = "h-[18px] w-[18px] shrink-0";
  const weight = "bold";
  switch (icon) {
    case IconType.Edit:
      return <UserIcon className={className} weight={weight} />;
    case IconType.LayoutDashboard:
      return <ChartLineUpIcon className={className} weight={weight} />;
    case IconType.UploadCloud:
      return <UploadSimpleIcon className={className} weight={weight} />;
    case IconType.Sliders:
      return <SlidersHorizontalIcon className={className} weight={weight} />;
    case IconType.Settings:
      return <GearIcon className={className} weight={weight} />;
    case IconType.Billing:
      return <CreditCardIcon className={className} weight={weight} />;
    case IconType.Help:
      return <QuestionIcon className={className} weight={weight} />;
    case IconType.LogOut:
      return <SignOutIcon className={className} weight={weight} />;
    default:
      return null;
  }
}
