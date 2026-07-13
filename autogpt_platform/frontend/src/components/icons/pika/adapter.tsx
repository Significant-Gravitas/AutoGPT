"use client";

/**
 * Flag-gated icon adapter. Each export renders the Phosphor icon by
 * default, or the mapped Pika (stroke) icon when Flag.PIKA_ICONS is on.
 * Call sites keep importing the same Phosphor names; only the import
 * source is swapped to this module. Generated from .context/pika/mapping.json.
 */
import React from "react";
import type { Icon, IconProps } from "@phosphor-icons/react";
import {
  Alien as PhAlien,
  AppWindowIcon as PhAppWindowIcon,
  ArrowBendLeftUpIcon as PhArrowBendLeftUpIcon,
  ArrowBendRightDownIcon as PhArrowBendRightDownIcon,
  ArrowBendUpRight as PhArrowBendUpRight,
  ArrowClockwise as PhArrowClockwise,
  ArrowCounterClockwise as PhArrowCounterClockwise,
  ArrowDownIcon as PhArrowDownIcon,
  ArrowLeft as PhArrowLeft,
  ArrowLeftIcon as PhArrowLeftIcon,
  ArrowRight as PhArrowRight,
  ArrowRightIcon as PhArrowRightIcon,
  ArrowSquareOut as PhArrowSquareOut,
  ArrowSquareOutIcon as PhArrowSquareOutIcon,
  ArrowUUpLeftIcon as PhArrowUUpLeftIcon,
  ArrowUUpRightIcon as PhArrowUUpRightIcon,
  ArrowUp as PhArrowUp,
  ArrowUpIcon as PhArrowUpIcon,
  ArrowsClockwiseIcon as PhArrowsClockwiseIcon,
  ArrowsOutIcon as PhArrowsOutIcon,
  ArrowsOutSimpleIcon as PhArrowsOutSimpleIcon,
  ArticleIcon as PhArticleIcon,
  Bell as PhBell,
  BellRinging as PhBellRinging,
  BellRingingIcon as PhBellRingingIcon,
  BellSimpleRingingIcon as PhBellSimpleRingingIcon,
  BellSlash as PhBellSlash,
  BookOpenIcon as PhBookOpenIcon,
  Books as PhBooks,
  BracketsCurlyIcon as PhBracketsCurlyIcon,
  Brain as PhBrain,
  BrainIcon as PhBrainIcon,
  Bug as PhBug,
  CalculatorIcon as PhCalculatorIcon,
  CalendarDotsIcon as PhCalendarDotsIcon,
  CameraIcon as PhCameraIcon,
  CardsThreeIcon as PhCardsThreeIcon,
  CaretDown as PhCaretDown,
  CaretDownIcon as PhCaretDownIcon,
  CaretLeft as PhCaretLeft,
  CaretLeftIcon as PhCaretLeftIcon,
  CaretRight as PhCaretRight,
  CaretRightIcon as PhCaretRightIcon,
  CaretUpDownIcon as PhCaretUpDownIcon,
  CaretUpIcon as PhCaretUpIcon,
  ChalkboardIcon as PhChalkboardIcon,
  ChartBarIcon as PhChartBarIcon,
  ChartLineIcon as PhChartLineIcon,
  ChartLineUpIcon as PhChartLineUpIcon,
  ChatCircle as PhChatCircle,
  ChatCircleDotsIcon as PhChatCircleDotsIcon,
  ChatCircleIcon as PhChatCircleIcon,
  ChatCircleTextIcon as PhChatCircleTextIcon,
  ChatTeardropDotsIcon as PhChatTeardropDotsIcon,
  ChatsCircleIcon as PhChatsCircleIcon,
  ChatsIcon as PhChatsIcon,
  Check as PhCheck,
  CheckCircle as PhCheckCircle,
  CheckCircleIcon as PhCheckCircleIcon,
  CheckIcon as PhCheckIcon,
  CheckSquareIcon as PhCheckSquareIcon,
  CircleDashedIcon as PhCircleDashedIcon,
  CircleIcon as PhCircleIcon,
  CircleNotch as PhCircleNotch,
  CircleNotchIcon as PhCircleNotchIcon,
  Clock as PhClock,
  ClockClockwiseIcon as PhClockClockwiseIcon,
  ClockCountdownIcon as PhClockCountdownIcon,
  ClockCounterClockwiseIcon as PhClockCounterClockwiseIcon,
  ClockIcon as PhClockIcon,
  CloudArrowUp as PhCloudArrowUp,
  Code as PhCode,
  CodeIcon as PhCodeIcon,
  CoinIcon as PhCoinIcon,
  CoinsIcon as PhCoinsIcon,
  CookieIcon as PhCookieIcon,
  Copy as PhCopy,
  CopyIcon as PhCopyIcon,
  CopySimple as PhCopySimple,
  Cpu as PhCpu,
  CreditCard as PhCreditCard,
  CreditCardIcon as PhCreditCardIcon,
  Cube as PhCube,
  CubeIcon as PhCubeIcon,
  CurrencyDollar as PhCurrencyDollar,
  CurrencyDollarSimpleIcon as PhCurrencyDollarSimpleIcon,
  DeviceMobile as PhDeviceMobile,
  DeviceMobileIcon as PhDeviceMobileIcon,
  DiscordLogo as PhDiscordLogo,
  DiscordLogoIcon as PhDiscordLogoIcon,
  DotsNineIcon as PhDotsNineIcon,
  DotsThree as PhDotsThree,
  DotsThreeIcon as PhDotsThreeIcon,
  DotsThreeOutlineVerticalIcon as PhDotsThreeOutlineVerticalIcon,
  DotsThreeVertical as PhDotsThreeVertical,
  DotsThreeVerticalIcon as PhDotsThreeVerticalIcon,
  Download as PhDownload,
  DownloadIcon as PhDownloadIcon,
  DownloadSimple as PhDownloadSimple,
  DownloadSimpleIcon as PhDownloadSimpleIcon,
  EarIcon as PhEarIcon,
  EnvelopeSimpleIcon as PhEnvelopeSimpleIcon,
  Eye as PhEye,
  EyeClosedIcon as PhEyeClosedIcon,
  EyeIcon as PhEyeIcon,
  EyeSlash as PhEyeSlash,
  FacebookLogo as PhFacebookLogo,
  File as PhFile,
  FileHtml as PhFileHtml,
  FileIcon as PhFileIcon,
  FileMagnifyingGlassIcon as PhFileMagnifyingGlassIcon,
  FilePdfIcon as PhFilePdfIcon,
  FilePlusIcon as PhFilePlusIcon,
  FileText as PhFileText,
  FileTextIcon as PhFileTextIcon,
  FilesIcon as PhFilesIcon,
  Flask as PhFlask,
  FlaskIcon as PhFlaskIcon,
  FloppyDisk as PhFloppyDisk,
  FloppyDiskIcon as PhFloppyDiskIcon,
  FlowArrow as PhFlowArrow,
  FlowArrowIcon as PhFlowArrowIcon,
  Folder as PhFolder,
  FolderIcon as PhFolderIcon,
  FolderOpen as PhFolderOpen,
  FolderOpenIcon as PhFolderOpenIcon,
  FolderPlusIcon as PhFolderPlusIcon,
  FolderSimpleIcon as PhFolderSimpleIcon,
  FolderSimplePlusIcon as PhFolderSimplePlusIcon,
  FoldersIcon as PhFoldersIcon,
  FrameCornersIcon as PhFrameCornersIcon,
  FunnelIcon as PhFunnelIcon,
  Gauge as PhGauge,
  GaugeIcon as PhGaugeIcon,
  Gear as PhGear,
  GearIcon as PhGearIcon,
  GearSix as PhGearSix,
  GithubLogo as PhGithubLogo,
  GithubLogoIcon as PhGithubLogoIcon,
  Globe as PhGlobe,
  GlobeIcon as PhGlobeIcon,
  GlobeSimple as PhGlobeSimple,
  GoogleLogoIcon as PhGoogleLogoIcon,
  HammerIcon as PhHammerIcon,
  HeadsetIcon as PhHeadsetIcon,
  Heart as PhHeart,
  HeartIcon as PhHeartIcon,
  Heartbeat as PhHeartbeat,
  HourglassIcon as PhHourglassIcon,
  HouseIcon as PhHouseIcon,
  Image as PhImage,
  ImageBroken as PhImageBroken,
  ImageBrokenIcon as PhImageBrokenIcon,
  ImageIcon as PhImageIcon,
  ImagesIcon as PhImagesIcon,
  Info as PhInfo,
  InfoIcon as PhInfoIcon,
  InstagramLogo as PhInstagramLogo,
  Key as PhKey,
  KeyIcon as PhKeyIcon,
  KeyholeIcon as PhKeyholeIcon,
  Laptop as PhLaptop,
  LegoIcon as PhLegoIcon,
  LightbulbIcon as PhLightbulbIcon,
  Lightning as PhLightning,
  LightningIcon as PhLightningIcon,
  LinkBreak as PhLinkBreak,
  LinkIcon as PhLinkIcon,
  LinkSimpleIcon as PhLinkSimpleIcon,
  LinkedinLogo as PhLinkedinLogo,
  List as PhList,
  ListBulletsIcon as PhListBulletsIcon,
  ListChecksIcon as PhListChecksIcon,
  ListIcon as PhListIcon,
  Lock as PhLock,
  MagicWandIcon as PhMagicWandIcon,
  MagnifyingGlass as PhMagnifyingGlass,
  MagnifyingGlassIcon as PhMagnifyingGlassIcon,
  MediumLogoIcon as PhMediumLogoIcon,
  MicrophoneIcon as PhMicrophoneIcon,
  MinusCircleIcon as PhMinusCircleIcon,
  MinusIcon as PhMinusIcon,
  MonitorIcon as PhMonitorIcon,
  MonitorPlayIcon as PhMonitorPlayIcon,
  MoonIcon as PhMoonIcon,
  NotePencil as PhNotePencil,
  NotePencilIcon as PhNotePencilIcon,
  NotionLogoIcon as PhNotionLogoIcon,
  Package as PhPackage,
  PackageIcon as PhPackageIcon,
  PaintBrushIcon as PhPaintBrushIcon,
  Password as PhPassword,
  PauseCircleIcon as PhPauseCircleIcon,
  PencilIcon as PhPencilIcon,
  PencilLineIcon as PhPencilLineIcon,
  PencilSimple as PhPencilSimple,
  PencilSimpleIcon as PhPencilSimpleIcon,
  PhoneCallIcon as PhPhoneCallIcon,
  PhoneIcon as PhPhoneIcon,
  Play as PhPlay,
  PlayCircleIcon as PhPlayCircleIcon,
  PlayIcon as PhPlayIcon,
  PlugIcon as PhPlugIcon,
  PlugsConnectedIcon as PhPlugsConnectedIcon,
  PlugsIcon as PhPlugsIcon,
  Plus as PhPlus,
  PlusCircleIcon as PhPlusCircleIcon,
  PlusIcon as PhPlusIcon,
  PowerIcon as PhPowerIcon,
  ProhibitIcon as PhProhibitIcon,
  Pulse as PhPulse,
  PushPinIcon as PhPushPinIcon,
  PushPinSlashIcon as PhPushPinSlashIcon,
  QuestionIcon as PhQuestionIcon,
  Receipt as PhReceipt,
  RobotIcon as PhRobotIcon,
  RocketLaunchIcon as PhRocketLaunchIcon,
  SealCheckIcon as PhSealCheckIcon,
  ShareFatIcon as PhShareFatIcon,
  ShareIcon as PhShareIcon,
  ShareNetworkIcon as PhShareNetworkIcon,
  ShieldCheckIcon as PhShieldCheckIcon,
  ShieldIcon as PhShieldIcon,
  SidebarSimpleIcon as PhSidebarSimpleIcon,
  SignInIcon as PhSignInIcon,
  SignOut as PhSignOut,
  SignOutIcon as PhSignOutIcon,
  SlidersHorizontalIcon as PhSlidersHorizontalIcon,
  SmileySadIcon as PhSmileySadIcon,
  SparkleIcon as PhSparkleIcon,
  SpeakerHigh as PhSpeakerHigh,
  Spinner as PhSpinner,
  SpinnerGapIcon as PhSpinnerGapIcon,
  SpinnerIcon as PhSpinnerIcon,
  SquareIcon as PhSquareIcon,
  SquaresFour as PhSquaresFour,
  SquaresFourIcon as PhSquaresFourIcon,
  StackIcon as PhStackIcon,
  Star as PhStar,
  StarIcon as PhStarIcon,
  Stop as PhStop,
  StopCircleIcon as PhStopCircleIcon,
  StopIcon as PhStopIcon,
  StorefrontIcon as PhStorefrontIcon,
  SunIcon as PhSunIcon,
  Table as PhTable,
  TableIcon as PhTableIcon,
  TerminalIcon as PhTerminalIcon,
  TextBIcon as PhTextBIcon,
  TextItalicIcon as PhTextItalicIcon,
  TextStrikethroughIcon as PhTextStrikethroughIcon,
  ThumbsDown as PhThumbsDown,
  ThumbsUp as PhThumbsUp,
  TiktokLogo as PhTiktokLogo,
  Trash as PhTrash,
  TrashIcon as PhTrashIcon,
  Tray as PhTray,
  TreeStructureIcon as PhTreeStructureIcon,
  TwitterLogoIcon as PhTwitterLogoIcon,
  UploadIcon as PhUploadIcon,
  UploadSimple as PhUploadSimple,
  UploadSimpleIcon as PhUploadSimpleIcon,
  User as PhUser,
  UserCheck as PhUserCheck,
  UserCircle as PhUserCircle,
  UserCircleDashedIcon as PhUserCircleDashedIcon,
  UserCircleIcon as PhUserCircleIcon,
  UserIcon as PhUserIcon,
  UserMinus as PhUserMinus,
  UserPlus as PhUserPlus,
  Users as PhUsers,
  UsersIcon as PhUsersIcon,
  VideoCamera as PhVideoCamera,
  VideoCameraIcon as PhVideoCameraIcon,
  WalletIcon as PhWalletIcon,
  Warning as PhWarning,
  WarningCircle as PhWarningCircle,
  WarningCircleIcon as PhWarningCircleIcon,
  WarningDiamondIcon as PhWarningDiamondIcon,
  WarningIcon as PhWarningIcon,
  WarningOctagonIcon as PhWarningOctagonIcon,
  WebhooksLogoIcon as PhWebhooksLogoIcon,
  X as PhX,
  XCircleIcon as PhXCircleIcon,
  XIcon as PhXIcon,
  XLogo as PhXLogo,
  YoutubeLogo as PhYoutubeLogo,
} from "@phosphor-icons/react";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import PiActivityStroke from "./vendor/PiActivityStroke";
import PiAlertCircleStroke from "./vendor/PiAlertCircleStroke";
import PiAlertTriangleStroke from "./vendor/PiAlertTriangleStroke";
import PiArrowDownStroke from "./vendor/PiArrowDownStroke";
import PiArrowLeftStroke from "./vendor/PiArrowLeftStroke";
import PiArrowRightStroke from "./vendor/PiArrowRightStroke";
import PiArrowTurnLeftUpStroke from "./vendor/PiArrowTurnLeftUpStroke";
import PiArrowTurnRightDownStroke from "./vendor/PiArrowTurnRightDownStroke";
import PiArrowTurnUpRightStroke from "./vendor/PiArrowTurnUpRightStroke";
import PiArrowUpStroke from "./vendor/PiArrowUpStroke";
import PiAtomStroke from "./vendor/PiAtomStroke";
import PiAutomationStroke from "./vendor/PiAutomationStroke";
import PiBanRightStroke from "./vendor/PiBanRightStroke";
import PiBarchartDefaultStroke from "./vendor/PiBarchartDefaultStroke";
import PiBoldStroke from "./vendor/PiBoldStroke";
import PiBotStroke from "./vendor/PiBotStroke";
import PiBugStroke from "./vendor/PiBugStroke";
import PiBulbDefaultStroke from "./vendor/PiBulbDefaultStroke";
import PiCalendarDefaultStroke from "./vendor/PiCalendarDefaultStroke";
import PiCameraDefaultStroke from "./vendor/PiCameraDefaultStroke";
import PiChatChattingStroke from "./vendor/PiChatChattingStroke";
import PiChatPlusStroke from "./vendor/PiChatPlusStroke";
import PiChatDefaultStroke from "./vendor/PiChatDefaultStroke";
import PiChatTypingStroke from "./vendor/PiChatTypingStroke";
import PiCheckTickCircleStroke from "./vendor/PiCheckTickCircleStroke";
import PiCheckTickSingleStroke from "./vendor/PiCheckTickSingleStroke";
import PiCheckTickSquareStroke from "./vendor/PiCheckTickSquareStroke";
import PiChevronDownStroke from "./vendor/PiChevronDownStroke";
import PiChevronLeftStroke from "./vendor/PiChevronLeftStroke";
import PiChevronRightStroke from "./vendor/PiChevronRightStroke";
import PiChevronSortVerticalStroke from "./vendor/PiChevronSortVerticalStroke";
import PiChevronUpStroke from "./vendor/PiChevronUpStroke";
import PiCircleDashedStroke from "./vendor/PiCircleDashedStroke";
import PiCircleStroke from "./vendor/PiCircleStroke";
import PiClockDefaultStroke from "./vendor/PiClockDefaultStroke";
import PiCloudArrowUploadStroke from "./vendor/PiCloudArrowUploadStroke";
import PiCodeStroke from "./vendor/PiCodeStroke";
import PiCookieStroke from "./vendor/PiCookieStroke";
import PiCopyDefaultStroke from "./vendor/PiCopyDefaultStroke";
import PiCreditCardStroke from "./vendor/PiCreditCardStroke";
import PiCubeStroke from "./vendor/PiCubeStroke";
import PiCurlyBracesCodeDefaultStroke from "./vendor/PiCurlyBracesCodeDefaultStroke";
import PiCurrencySignDollarStroke from "./vendor/PiCurrencySignDollarStroke";
import PiDeleteDustbin01Stroke from "./vendor/PiDeleteDustbin01Stroke";
import PiDiscordStroke from "./vendor/PiDiscordStroke";
import PiDownloadDownStroke from "./vendor/PiDownloadDownStroke";
import PiEarStroke from "./vendor/PiEarStroke";
import PiEnvelopeDefaultStroke from "./vendor/PiEnvelopeDefaultStroke";
import PiExternalLinkSquareStroke from "./vendor/PiExternalLinkSquareStroke";
import PiEyeOffStroke from "./vendor/PiEyeOffStroke";
import PiEyeOnStroke from "./vendor/PiEyeOnStroke";
import PiFaceSadStroke from "./vendor/PiFaceSadStroke";
import PiFacebookStroke from "./vendor/PiFacebookStroke";
import PiFeatherPenStroke from "./vendor/PiFeatherPenStroke";
import PiFileCodeStroke from "./vendor/PiFileCodeStroke";
import PiFileDefaultStroke from "./vendor/PiFileDefaultStroke";
import PiFilePdfFormatStroke from "./vendor/PiFilePdfFormatStroke";
import PiFilePlusStroke from "./vendor/PiFilePlusStroke";
import PiFileSearchStroke from "./vendor/PiFileSearchStroke";
import PiFileTextStroke from "./vendor/PiFileTextStroke";
import PiFilterFunnelStroke from "./vendor/PiFilterFunnelStroke";
import PiFilterHorizontalStroke from "./vendor/PiFilterHorizontalStroke";
import PiFloppyDefaultStroke from "./vendor/PiFloppyDefaultStroke";
import PiFolderDefaultStroke from "./vendor/PiFolderDefaultStroke";
import PiFolderOpenStroke from "./vendor/PiFolderOpenStroke";
import PiFolderPlusStroke from "./vendor/PiFolderPlusStroke";
import PiFrameStroke from "./vendor/PiFrameStroke";
import PiGaugeSpeedometerStroke from "./vendor/PiGaugeSpeedometerStroke";
import PiGitBranchStroke from "./vendor/PiGitBranchStroke";
import PiGithubStroke from "./vendor/PiGithubStroke";
import PiGlobeStroke from "./vendor/PiGlobeStroke";
import PiGoogleStroke from "./vendor/PiGoogleStroke";
import PiGraphChartLineStroke from "./vendor/PiGraphChartLineStroke";
import PiGraphTrendLineUpwardStroke from "./vendor/PiGraphTrendLineUpwardStroke";
import PiGrid01Stroke from "./vendor/PiGrid01Stroke";
import PiGridTableStroke from "./vendor/PiGridTableStroke";
import PiHeadphonesStroke from "./vendor/PiHeadphonesStroke";
import PiHeartStroke from "./vendor/PiHeartStroke";
import PiHomeDefaultStroke from "./vendor/PiHomeDefaultStroke";
import PiHourglassStroke from "./vendor/PiHourglassStroke";
import PiInboxDefaultStroke from "./vendor/PiInboxDefaultStroke";
import PiInformationCircleStroke from "./vendor/PiInformationCircleStroke";
import PiInstagramStroke from "./vendor/PiInstagramStroke";
import PiIphoneStroke from "./vendor/PiIphoneStroke";
import PiItalicStroke from "./vendor/PiItalicStroke";
import PiKeyLeftStroke from "./vendor/PiKeyLeftStroke";
import PiLabFlaskConicalStroke from "./vendor/PiLabFlaskConicalStroke";
import PiLaptopStroke from "./vendor/PiLaptopStroke";
import PiLayersToStroke from "./vendor/PiLayersToStroke";
import PiLayoutGridTwoVerticalStroke from "./vendor/PiLayoutGridTwoVerticalStroke";
import PiLibraryStroke from "./vendor/PiLibraryStroke";
import PiLightningThunderElectricOnStroke from "./vendor/PiLightningThunderElectricOnStroke";
import PiLinkHorizontalBrokenStroke from "./vendor/PiLinkHorizontalBrokenStroke";
import PiLinkHorizontalStroke from "./vendor/PiLinkHorizontalStroke";
import PiLinkedinStroke from "./vendor/PiLinkedinStroke";
import PiListCheckStroke from "./vendor/PiListCheckStroke";
import PiListDefaultStroke from "./vendor/PiListDefaultStroke";
import PiLockCloseStroke from "./vendor/PiLockCloseStroke";
import PiLogInRightStroke from "./vendor/PiLogInRightStroke";
import PiLogOutRightStroke from "./vendor/PiLogOutRightStroke";
import PiMagicWandStroke from "./vendor/PiMagicWandStroke";
import PiMathStroke from "./vendor/PiMathStroke";
import PiMaximizeFourArrowStroke from "./vendor/PiMaximizeFourArrowStroke";
import PiMaximizeLineArrowStroke from "./vendor/PiMaximizeLineArrowStroke";
import PiMediaPlaySquareStroke from "./vendor/PiMediaPlaySquareStroke";
import PiMicMicrophoneStroke from "./vendor/PiMicMicrophoneStroke";
import PiMinusCircleStroke from "./vendor/PiMinusCircleStroke";
import PiMinusDefaultStroke from "./vendor/PiMinusDefaultStroke";
import PiMoneyDollarBagStroke from "./vendor/PiMoneyDollarBagStroke";
import PiMonitor01Stroke from "./vendor/PiMonitor01Stroke";
import PiMoonStroke from "./vendor/PiMoonStroke";
import PiMultipleCrossCancelCircleStroke from "./vendor/PiMultipleCrossCancelCircleStroke";
import PiMultipleCrossCancelDefaultStroke from "./vendor/PiMultipleCrossCancelDefaultStroke";
import PiNotebookStroke from "./vendor/PiNotebookStroke";
import PiNotificationBellOffStroke from "./vendor/PiNotificationBellOffStroke";
import PiNotificationBellOnStroke from "./vendor/PiNotificationBellOnStroke";
import PiNotionStroke from "./vendor/PiNotionStroke";
import PiPackage01Stroke from "./vendor/PiPackage01Stroke";
import PiPaintBrushStroke from "./vendor/PiPaintBrushStroke";
import PiPauseCircleStroke from "./vendor/PiPauseCircleStroke";
import PiPencilEditBoxStroke from "./vendor/PiPencilEditBoxStroke";
import PiPencilEditLineStroke from "./vendor/PiPencilEditLineStroke";
import PiPencilEditStroke from "./vendor/PiPencilEditStroke";
import PiPhoneDefaultStroke from "./vendor/PiPhoneDefaultStroke";
import PiPhoneOutgoingStroke from "./vendor/PiPhoneOutgoingStroke";
import PiPhotoImageDefaultStroke from "./vendor/PiPhotoImageDefaultStroke";
import PiPhotoImageRemoveStroke from "./vendor/PiPhotoImageRemoveStroke";
import PiPinDefaultStroke from "./vendor/PiPinDefaultStroke";
import PiPinSlantStroke from "./vendor/PiPinSlantStroke";
import PiPlayBigStroke from "./vendor/PiPlayBigStroke";
import PiPlayCircleStroke from "./vendor/PiPlayCircleStroke";
import PiPluginAddonDefaultStroke from "./vendor/PiPluginAddonDefaultStroke";
import PiPluginAddonPuzzleStroke from "./vendor/PiPluginAddonPuzzleStroke";
import PiPlusCircleStroke from "./vendor/PiPlusCircleStroke";
import PiPlusDefaultStroke from "./vendor/PiPlusDefaultStroke";
import PiPowerDefaultStroke from "./vendor/PiPowerDefaultStroke";
import PiPresentationBargraphStroke from "./vendor/PiPresentationBargraphStroke";
import PiQuestionMarkCircleStroke from "./vendor/PiQuestionMarkCircleStroke";
import PiReceipt01Stroke from "./vendor/PiReceipt01Stroke";
import PiRefreshStroke from "./vendor/PiRefreshStroke";
import PiReminderAnticlockwiseStroke from "./vendor/PiReminderAnticlockwiseStroke";
import PiReminderClockwiseStroke from "./vendor/PiReminderClockwiseStroke";
import PiRocketShipStroke from "./vendor/PiRocketShipStroke";
import PiRotateLeftStroke from "./vendor/PiRotateLeftStroke";
import PiRotateRightStroke from "./vendor/PiRotateRightStroke";
import PiSearchDefaultStroke from "./vendor/PiSearchDefaultStroke";
import PiSettings01Stroke from "./vendor/PiSettings01Stroke";
import PiShare01Stroke from "./vendor/PiShare01Stroke";
import PiShare02Stroke from "./vendor/PiShare02Stroke";
import PiShieldCheckStroke from "./vendor/PiShieldCheckStroke";
import PiShieldStroke from "./vendor/PiShieldStroke";
import PiSidebarDefaultStroke from "./vendor/PiSidebarDefaultStroke";
import PiSparkleAI01Stroke from "./vendor/PiSparkleAI01Stroke";
import PiSpeakerOnStroke from "./vendor/PiSpeakerOnStroke";
import PiSpinnerStroke from "./vendor/PiSpinnerStroke";
import PiSquareStroke from "./vendor/PiSquareStroke";
import PiStarStroke from "./vendor/PiStarStroke";
import PiStopBigStroke from "./vendor/PiStopBigStroke";
import PiStopCircleStroke from "./vendor/PiStopCircleStroke";
import PiStopSquareStroke from "./vendor/PiStopSquareStroke";
import PiStoreDefaultStroke from "./vendor/PiStoreDefaultStroke";
import PiStrikeThroughStroke from "./vendor/PiStrikeThroughStroke";
import PiSunStroke from "./vendor/PiSunStroke";
import PiTerminalConsoleSquareStroke from "./vendor/PiTerminalConsoleSquareStroke";
import PiThreeDotsMenuHorizontalStroke from "./vendor/PiThreeDotsMenuHorizontalStroke";
import PiThreeDotsMenuVerticalStroke from "./vendor/PiThreeDotsMenuVerticalStroke";
import PiThumbReactionDislikeStroke from "./vendor/PiThumbReactionDislikeStroke";
import PiThumbReactionLikeStroke from "./vendor/PiThumbReactionLikeStroke";
import PiTiktokStroke from "./vendor/PiTiktokStroke";
import PiTimerDefaultStroke from "./vendor/PiTimerDefaultStroke";
import PiTokenStroke from "./vendor/PiTokenStroke";
import PiToolsStroke from "./vendor/PiToolsStroke";
import PiTwitterStroke from "./vendor/PiTwitterStroke";
import PiUfoStroke from "./vendor/PiUfoStroke";
import PiUploadUpStroke from "./vendor/PiUploadUpStroke";
import PiUserCheckStroke from "./vendor/PiUserCheckStroke";
import PiUserCircleDottedStroke from "./vendor/PiUserCircleDottedStroke";
import PiUserCircleStroke from "./vendor/PiUserCircleStroke";
import PiUserDefaultStroke from "./vendor/PiUserDefaultStroke";
import PiUserPlusStroke from "./vendor/PiUserPlusStroke";
import PiUserRemoveStroke from "./vendor/PiUserRemoveStroke";
import PiUserTwoStroke from "./vendor/PiUserTwoStroke";
import PiUturnLeftStroke from "./vendor/PiUturnLeftStroke";
import PiUturnRightStroke from "./vendor/PiUturnRightStroke";
import PiVerificationCheckStroke from "./vendor/PiVerificationCheckStroke";
import PiVideoRecordingStroke from "./vendor/PiVideoRecordingStroke";
import PiWalletDefaultStroke from "./vendor/PiWalletDefaultStroke";
import PiWebhookStroke from "./vendor/PiWebhookStroke";
import PiWindowBrowserStroke from "./vendor/PiWindowBrowserStroke";
import PiXComStroke from "./vendor/PiXComStroke";
import PiYoutubeStroke from "./vendor/PiYoutubeStroke";

interface PikaIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

function makeIcon(
  PhosphorIcon: Icon,
  PikaIcon: React.ComponentType<PikaIconProps>,
): Icon {
  return React.forwardRef<SVGSVGElement, IconProps>(
    function AdaptiveIcon(props, ref) {
      const usePika = useGetFlag(Flag.PIKA_ICONS);
      if (!usePika) return <PhosphorIcon ref={ref} {...props} />;
      const {
        weight: _weight,
        mirrored: _mirrored,
        alt,
        size,
        color,
        className,
        style,
        ...rest
      } = props;
      const numericSize =
        typeof size === "number"
          ? size
          : size != null
            ? Number.parseFloat(String(size))
            : undefined;
      // Pika vendor icons hardcode role="img" + a default aria-label, which
      // makes otherwise-decorative icons announced content and pollutes the
      // accessible name of any labeled control they sit inside. Phosphor icons
      // are unlabeled by default, so mirror that: strip role/aria-label unless
      // the caller passes `alt` (Phosphor's opt-in labeling prop).
      const a11yProps = alt
        ? { role: "img", "aria-label": alt }
        : { role: undefined, "aria-label": undefined };
      return (
        <PikaIcon
          size={
            numericSize != null && !Number.isNaN(numericSize)
              ? numericSize
              : undefined
          }
          color={color}
          className={className}
          style={style}
          {...a11yProps}
          {...(rest as PikaIconProps)}
        />
      );
    },
  ) as Icon;
}

export type { Icon, IconProps };

export const Alien = makeIcon(PhAlien, PiUfoStroke);
export const AppWindowIcon = makeIcon(PhAppWindowIcon, PiWindowBrowserStroke);
export const ArrowBendLeftUpIcon = makeIcon(
  PhArrowBendLeftUpIcon,
  PiArrowTurnLeftUpStroke,
);
export const ArrowBendRightDownIcon = makeIcon(
  PhArrowBendRightDownIcon,
  PiArrowTurnRightDownStroke,
);
export const ArrowBendUpRight = makeIcon(
  PhArrowBendUpRight,
  PiArrowTurnUpRightStroke,
);
export const ArrowClockwise = makeIcon(PhArrowClockwise, PiRotateRightStroke);
export const ArrowCounterClockwise = makeIcon(
  PhArrowCounterClockwise,
  PiRotateLeftStroke,
);
export const ArrowDownIcon = makeIcon(PhArrowDownIcon, PiArrowDownStroke);
export const ArrowLeft = makeIcon(PhArrowLeft, PiArrowLeftStroke);
export const ArrowLeftIcon = makeIcon(PhArrowLeftIcon, PiArrowLeftStroke);
export const ArrowRight = makeIcon(PhArrowRight, PiArrowRightStroke);
export const ArrowRightIcon = makeIcon(PhArrowRightIcon, PiArrowRightStroke);
export const ArrowSquareOut = makeIcon(
  PhArrowSquareOut,
  PiExternalLinkSquareStroke,
);
export const ArrowSquareOutIcon = makeIcon(
  PhArrowSquareOutIcon,
  PiExternalLinkSquareStroke,
);
export const ArrowUUpLeftIcon = makeIcon(PhArrowUUpLeftIcon, PiUturnLeftStroke);
export const ArrowUUpRightIcon = makeIcon(
  PhArrowUUpRightIcon,
  PiUturnRightStroke,
);
export const ArrowUp = makeIcon(PhArrowUp, PiArrowUpStroke);
export const ArrowUpIcon = makeIcon(PhArrowUpIcon, PiArrowUpStroke);
export const ArrowsClockwiseIcon = makeIcon(
  PhArrowsClockwiseIcon,
  PiRefreshStroke,
);
export const ArrowsOutIcon = makeIcon(
  PhArrowsOutIcon,
  PiMaximizeFourArrowStroke,
);
export const ArrowsOutSimpleIcon = makeIcon(
  PhArrowsOutSimpleIcon,
  PiMaximizeLineArrowStroke,
);
export const ArticleIcon = makeIcon(PhArticleIcon, PiFileTextStroke);
export const Bell = makeIcon(PhBell, PiNotificationBellOnStroke);
export const BellRinging = makeIcon(PhBellRinging, PiNotificationBellOnStroke);
export const BellRingingIcon = makeIcon(
  PhBellRingingIcon,
  PiNotificationBellOnStroke,
);
export const BellSimpleRingingIcon = makeIcon(
  PhBellSimpleRingingIcon,
  PiNotificationBellOnStroke,
);
export const BellSlash = makeIcon(PhBellSlash, PiNotificationBellOffStroke);
export const BookOpenIcon = makeIcon(PhBookOpenIcon, PiNotebookStroke);
export const Books = makeIcon(PhBooks, PiLibraryStroke);
export const BracketsCurlyIcon = makeIcon(
  PhBracketsCurlyIcon,
  PiCurlyBracesCodeDefaultStroke,
);
export const Brain = makeIcon(PhBrain, PiAtomStroke);
export const BrainIcon = makeIcon(PhBrainIcon, PiAtomStroke);
export const Bug = makeIcon(PhBug, PiBugStroke);
export const CalculatorIcon = makeIcon(PhCalculatorIcon, PiMathStroke);
export const CalendarDotsIcon = makeIcon(
  PhCalendarDotsIcon,
  PiCalendarDefaultStroke,
);
export const CameraIcon = makeIcon(PhCameraIcon, PiCameraDefaultStroke);
export const CardsThreeIcon = makeIcon(
  PhCardsThreeIcon,
  PiLayoutGridTwoVerticalStroke,
);
export const CaretDown = makeIcon(PhCaretDown, PiChevronDownStroke);
export const CaretDownIcon = makeIcon(PhCaretDownIcon, PiChevronDownStroke);
export const CaretLeft = makeIcon(PhCaretLeft, PiChevronLeftStroke);
export const CaretLeftIcon = makeIcon(PhCaretLeftIcon, PiChevronLeftStroke);
export const CaretRight = makeIcon(PhCaretRight, PiChevronRightStroke);
export const CaretRightIcon = makeIcon(PhCaretRightIcon, PiChevronRightStroke);
export const CaretUpDownIcon = makeIcon(
  PhCaretUpDownIcon,
  PiChevronSortVerticalStroke,
);
export const CaretUpIcon = makeIcon(PhCaretUpIcon, PiChevronUpStroke);
export const ChalkboardIcon = makeIcon(
  PhChalkboardIcon,
  PiPresentationBargraphStroke,
);
export const ChartBarIcon = makeIcon(PhChartBarIcon, PiBarchartDefaultStroke);
export const ChartLineIcon = makeIcon(PhChartLineIcon, PiGraphChartLineStroke);
export const ChartLineUpIcon = makeIcon(
  PhChartLineUpIcon,
  PiGraphTrendLineUpwardStroke,
);
export const ChatCircle = makeIcon(PhChatCircle, PiChatDefaultStroke);
export const ChatCircleDotsIcon = makeIcon(
  PhChatCircleDotsIcon,
  PiChatTypingStroke,
);
export const ChatCircleIcon = makeIcon(PhChatCircleIcon, PiChatDefaultStroke);
export const ChatCircleTextIcon = makeIcon(
  PhChatCircleTextIcon,
  PiChatDefaultStroke,
);
export const ChatTeardropDotsIcon = makeIcon(
  PhChatTeardropDotsIcon,
  PiChatTypingStroke,
);
export const ChatsCircleIcon = makeIcon(
  PhChatsCircleIcon,
  PiChatChattingStroke,
);
export const ChatsIcon = makeIcon(PhChatsIcon, PiChatChattingStroke);
export const Check = makeIcon(PhCheck, PiCheckTickSingleStroke);
export const CheckCircle = makeIcon(PhCheckCircle, PiCheckTickCircleStroke);
export const CheckCircleIcon = makeIcon(
  PhCheckCircleIcon,
  PiCheckTickCircleStroke,
);
export const CheckIcon = makeIcon(PhCheckIcon, PiCheckTickSingleStroke);
export const CheckSquareIcon = makeIcon(
  PhCheckSquareIcon,
  PiCheckTickSquareStroke,
);
export const CircleDashedIcon = makeIcon(
  PhCircleDashedIcon,
  PiCircleDashedStroke,
);
export const CircleIcon = makeIcon(PhCircleIcon, PiCircleStroke);
export const CircleNotch = makeIcon(PhCircleNotch, PiSpinnerStroke);
export const CircleNotchIcon = makeIcon(PhCircleNotchIcon, PiSpinnerStroke);
export const Clock = makeIcon(PhClock, PiClockDefaultStroke);
export const ClockClockwiseIcon = makeIcon(
  PhClockClockwiseIcon,
  PiReminderClockwiseStroke,
);
export const ClockCountdownIcon = makeIcon(
  PhClockCountdownIcon,
  PiTimerDefaultStroke,
);
export const ClockCounterClockwiseIcon = makeIcon(
  PhClockCounterClockwiseIcon,
  PiReminderAnticlockwiseStroke,
);
export const ClockIcon = makeIcon(PhClockIcon, PiClockDefaultStroke);
export const CloudArrowUp = makeIcon(PhCloudArrowUp, PiCloudArrowUploadStroke);
export const Code = makeIcon(PhCode, PiCodeStroke);
export const CodeIcon = makeIcon(PhCodeIcon, PiCodeStroke);
export const CoinIcon = makeIcon(PhCoinIcon, PiTokenStroke);
export const CoinsIcon = makeIcon(PhCoinsIcon, PiMoneyDollarBagStroke);
export const CookieIcon = makeIcon(PhCookieIcon, PiCookieStroke);
export const Copy = makeIcon(PhCopy, PiCopyDefaultStroke);
export const CopyIcon = makeIcon(PhCopyIcon, PiCopyDefaultStroke);
export const CopySimple = makeIcon(PhCopySimple, PiCopyDefaultStroke);
export const Cpu = makeIcon(PhCpu, PiTerminalConsoleSquareStroke);
export const CreditCard = makeIcon(PhCreditCard, PiCreditCardStroke);
export const CreditCardIcon = makeIcon(PhCreditCardIcon, PiCreditCardStroke);
export const Cube = makeIcon(PhCube, PiCubeStroke);
export const CubeIcon = makeIcon(PhCubeIcon, PiCubeStroke);
export const CurrencyDollar = makeIcon(
  PhCurrencyDollar,
  PiCurrencySignDollarStroke,
);
export const CurrencyDollarSimpleIcon = makeIcon(
  PhCurrencyDollarSimpleIcon,
  PiCurrencySignDollarStroke,
);
export const DeviceMobile = makeIcon(PhDeviceMobile, PiIphoneStroke);
export const DeviceMobileIcon = makeIcon(PhDeviceMobileIcon, PiIphoneStroke);
export const DiscordLogo = makeIcon(PhDiscordLogo, PiDiscordStroke);
export const DiscordLogoIcon = makeIcon(PhDiscordLogoIcon, PiDiscordStroke);
export const DotsNineIcon = makeIcon(PhDotsNineIcon, PiGrid01Stroke);
export const DotsThree = makeIcon(PhDotsThree, PiThreeDotsMenuHorizontalStroke);
export const DotsThreeIcon = makeIcon(
  PhDotsThreeIcon,
  PiThreeDotsMenuHorizontalStroke,
);
export const DotsThreeOutlineVerticalIcon = makeIcon(
  PhDotsThreeOutlineVerticalIcon,
  PiThreeDotsMenuVerticalStroke,
);
export const DotsThreeVertical = makeIcon(
  PhDotsThreeVertical,
  PiThreeDotsMenuVerticalStroke,
);
export const DotsThreeVerticalIcon = makeIcon(
  PhDotsThreeVerticalIcon,
  PiThreeDotsMenuVerticalStroke,
);
export const Download = makeIcon(PhDownload, PiDownloadDownStroke);
export const DownloadIcon = makeIcon(PhDownloadIcon, PiDownloadDownStroke);
export const DownloadSimple = makeIcon(PhDownloadSimple, PiDownloadDownStroke);
export const DownloadSimpleIcon = makeIcon(
  PhDownloadSimpleIcon,
  PiDownloadDownStroke,
);
export const EarIcon = makeIcon(PhEarIcon, PiEarStroke);
export const EnvelopeSimpleIcon = makeIcon(
  PhEnvelopeSimpleIcon,
  PiEnvelopeDefaultStroke,
);
export const Eye = makeIcon(PhEye, PiEyeOnStroke);
export const EyeClosedIcon = makeIcon(PhEyeClosedIcon, PiEyeOffStroke);
export const EyeIcon = makeIcon(PhEyeIcon, PiEyeOnStroke);
export const EyeSlash = makeIcon(PhEyeSlash, PiEyeOffStroke);
export const FacebookLogo = makeIcon(PhFacebookLogo, PiFacebookStroke);
export const File = makeIcon(PhFile, PiFileDefaultStroke);
export const FileHtml = makeIcon(PhFileHtml, PiFileCodeStroke);
export const FileIcon = makeIcon(PhFileIcon, PiFileDefaultStroke);
export const FileMagnifyingGlassIcon = makeIcon(
  PhFileMagnifyingGlassIcon,
  PiFileSearchStroke,
);
export const FilePdfIcon = makeIcon(PhFilePdfIcon, PiFilePdfFormatStroke);
export const FilePlusIcon = makeIcon(PhFilePlusIcon, PiFilePlusStroke);
export const FileText = makeIcon(PhFileText, PiFileTextStroke);
export const FileTextIcon = makeIcon(PhFileTextIcon, PiFileTextStroke);
export const FilesIcon = makeIcon(PhFilesIcon, PiFileDefaultStroke);
export const Flask = makeIcon(PhFlask, PiLabFlaskConicalStroke);
export const FlaskIcon = makeIcon(PhFlaskIcon, PiLabFlaskConicalStroke);
export const FloppyDisk = makeIcon(PhFloppyDisk, PiFloppyDefaultStroke);
export const FloppyDiskIcon = makeIcon(PhFloppyDiskIcon, PiFloppyDefaultStroke);
export const FlowArrow = makeIcon(PhFlowArrow, PiAutomationStroke);
export const FlowArrowIcon = makeIcon(PhFlowArrowIcon, PiAutomationStroke);
export const Folder = makeIcon(PhFolder, PiFolderDefaultStroke);
export const FolderIcon = makeIcon(PhFolderIcon, PiFolderDefaultStroke);
export const FolderOpen = makeIcon(PhFolderOpen, PiFolderOpenStroke);
export const FolderOpenIcon = makeIcon(PhFolderOpenIcon, PiFolderOpenStroke);
export const FolderPlusIcon = makeIcon(PhFolderPlusIcon, PiFolderPlusStroke);
export const FolderSimpleIcon = makeIcon(
  PhFolderSimpleIcon,
  PiFolderDefaultStroke,
);
export const FolderSimplePlusIcon = makeIcon(
  PhFolderSimplePlusIcon,
  PiFolderPlusStroke,
);
export const FoldersIcon = makeIcon(PhFoldersIcon, PiFolderDefaultStroke);
export const FrameCornersIcon = makeIcon(PhFrameCornersIcon, PiFrameStroke);
export const FunnelIcon = makeIcon(PhFunnelIcon, PiFilterFunnelStroke);
export const Gauge = makeIcon(PhGauge, PiGaugeSpeedometerStroke);
export const GaugeIcon = makeIcon(PhGaugeIcon, PiGaugeSpeedometerStroke);
export const Gear = makeIcon(PhGear, PiSettings01Stroke);
export const GearIcon = makeIcon(PhGearIcon, PiSettings01Stroke);
export const GearSix = makeIcon(PhGearSix, PiSettings01Stroke);
export const GithubLogo = makeIcon(PhGithubLogo, PiGithubStroke);
export const GithubLogoIcon = makeIcon(PhGithubLogoIcon, PiGithubStroke);
export const Globe = makeIcon(PhGlobe, PiGlobeStroke);
export const GlobeIcon = makeIcon(PhGlobeIcon, PiGlobeStroke);
export const GlobeSimple = makeIcon(PhGlobeSimple, PiGlobeStroke);
export const GoogleLogoIcon = makeIcon(PhGoogleLogoIcon, PiGoogleStroke);
export const HammerIcon = makeIcon(PhHammerIcon, PiToolsStroke);
export const HeadsetIcon = makeIcon(PhHeadsetIcon, PiHeadphonesStroke);
export const Heart = makeIcon(PhHeart, PiHeartStroke);
export const HeartIcon = makeIcon(PhHeartIcon, PiHeartStroke);
export const Heartbeat = makeIcon(PhHeartbeat, PiActivityStroke);
export const HourglassIcon = makeIcon(PhHourglassIcon, PiHourglassStroke);
export const HouseIcon = makeIcon(PhHouseIcon, PiHomeDefaultStroke);
export const Image = makeIcon(PhImage, PiPhotoImageDefaultStroke);
export const ImageBroken = makeIcon(PhImageBroken, PiPhotoImageRemoveStroke);
export const ImageBrokenIcon = makeIcon(
  PhImageBrokenIcon,
  PiPhotoImageRemoveStroke,
);
export const ImageIcon = makeIcon(PhImageIcon, PiPhotoImageDefaultStroke);
export const ImagesIcon = makeIcon(PhImagesIcon, PiPhotoImageDefaultStroke);
export const Info = makeIcon(PhInfo, PiInformationCircleStroke);
export const InfoIcon = makeIcon(PhInfoIcon, PiInformationCircleStroke);
export const InstagramLogo = makeIcon(PhInstagramLogo, PiInstagramStroke);
export const Key = makeIcon(PhKey, PiKeyLeftStroke);
export const KeyIcon = makeIcon(PhKeyIcon, PiKeyLeftStroke);
export const KeyholeIcon = makeIcon(PhKeyholeIcon, PiLockCloseStroke);
export const Laptop = makeIcon(PhLaptop, PiLaptopStroke);
export const LegoIcon = makeIcon(PhLegoIcon, PiPluginAddonPuzzleStroke);
export const LightbulbIcon = makeIcon(PhLightbulbIcon, PiBulbDefaultStroke);
export const Lightning = makeIcon(
  PhLightning,
  PiLightningThunderElectricOnStroke,
);
export const LightningIcon = makeIcon(
  PhLightningIcon,
  PiLightningThunderElectricOnStroke,
);
export const LinkBreak = makeIcon(PhLinkBreak, PiLinkHorizontalBrokenStroke);
export const LinkIcon = makeIcon(PhLinkIcon, PiLinkHorizontalStroke);
export const LinkSimpleIcon = makeIcon(
  PhLinkSimpleIcon,
  PiLinkHorizontalStroke,
);
export const LinkedinLogo = makeIcon(PhLinkedinLogo, PiLinkedinStroke);
export const List = makeIcon(PhList, PiListDefaultStroke);
export const ListBulletsIcon = makeIcon(PhListBulletsIcon, PiListDefaultStroke);
export const ListChecksIcon = makeIcon(PhListChecksIcon, PiListCheckStroke);
export const ListIcon = makeIcon(PhListIcon, PiListDefaultStroke);
export const Lock = makeIcon(PhLock, PiLockCloseStroke);
export const MagicWandIcon = makeIcon(PhMagicWandIcon, PiMagicWandStroke);
export const MagnifyingGlass = makeIcon(
  PhMagnifyingGlass,
  PiSearchDefaultStroke,
);
export const MagnifyingGlassIcon = makeIcon(
  PhMagnifyingGlassIcon,
  PiSearchDefaultStroke,
);
export const MediumLogoIcon = makeIcon(PhMediumLogoIcon, PiFeatherPenStroke);
export const MicrophoneIcon = makeIcon(PhMicrophoneIcon, PiMicMicrophoneStroke);
export const MinusCircleIcon = makeIcon(PhMinusCircleIcon, PiMinusCircleStroke);
export const MinusIcon = makeIcon(PhMinusIcon, PiMinusDefaultStroke);
export const MonitorIcon = makeIcon(PhMonitorIcon, PiMonitor01Stroke);
export const MonitorPlayIcon = makeIcon(
  PhMonitorPlayIcon,
  PiMediaPlaySquareStroke,
);
export const MoonIcon = makeIcon(PhMoonIcon, PiMoonStroke);
export const NotePencilIcon = makeIcon(PhNotePencilIcon, PiPencilEditBoxStroke);
// "New chat" compose icon: Phosphor has no chat-plus, fall back to NotePencil.
export const ChatPlusIcon = makeIcon(PhNotePencil, PiChatPlusStroke);
export const NotionLogoIcon = makeIcon(PhNotionLogoIcon, PiNotionStroke);
export const Package = makeIcon(PhPackage, PiPackage01Stroke);
export const PackageIcon = makeIcon(PhPackageIcon, PiPackage01Stroke);
export const PaintBrushIcon = makeIcon(PhPaintBrushIcon, PiPaintBrushStroke);
export const Password = makeIcon(PhPassword, PiLockCloseStroke);
export const PauseCircleIcon = makeIcon(PhPauseCircleIcon, PiPauseCircleStroke);
export const PencilIcon = makeIcon(PhPencilIcon, PiPencilEditStroke);
export const PencilLineIcon = makeIcon(
  PhPencilLineIcon,
  PiPencilEditLineStroke,
);
export const PencilSimple = makeIcon(PhPencilSimple, PiPencilEditStroke);
export const PencilSimpleIcon = makeIcon(
  PhPencilSimpleIcon,
  PiPencilEditStroke,
);
export const PhoneCallIcon = makeIcon(PhPhoneCallIcon, PiPhoneOutgoingStroke);
export const PhoneIcon = makeIcon(PhPhoneIcon, PiPhoneDefaultStroke);
export const Play = makeIcon(PhPlay, PiPlayBigStroke);
export const PlayCircleIcon = makeIcon(PhPlayCircleIcon, PiPlayCircleStroke);
export const PlayIcon = makeIcon(PhPlayIcon, PiPlayBigStroke);
export const PlugIcon = makeIcon(PhPlugIcon, PiPluginAddonDefaultStroke);
export const PlugsConnectedIcon = makeIcon(
  PhPlugsConnectedIcon,
  PiPluginAddonDefaultStroke,
);
export const PlugsIcon = makeIcon(PhPlugsIcon, PiPluginAddonDefaultStroke);
export const Plus = makeIcon(PhPlus, PiPlusDefaultStroke);
export const PlusCircleIcon = makeIcon(PhPlusCircleIcon, PiPlusCircleStroke);
export const PlusIcon = makeIcon(PhPlusIcon, PiPlusDefaultStroke);
export const PowerIcon = makeIcon(PhPowerIcon, PiPowerDefaultStroke);
export const ProhibitIcon = makeIcon(PhProhibitIcon, PiBanRightStroke);
export const Pulse = makeIcon(PhPulse, PiActivityStroke);
export const PushPinIcon = makeIcon(PhPushPinIcon, PiPinDefaultStroke);
export const PushPinSlashIcon = makeIcon(PhPushPinSlashIcon, PiPinSlantStroke);
export const QuestionIcon = makeIcon(
  PhQuestionIcon,
  PiQuestionMarkCircleStroke,
);
export const Receipt = makeIcon(PhReceipt, PiReceipt01Stroke);
export const RobotIcon = makeIcon(PhRobotIcon, PiBotStroke);
export const RocketLaunchIcon = makeIcon(
  PhRocketLaunchIcon,
  PiRocketShipStroke,
);
export const SealCheckIcon = makeIcon(
  PhSealCheckIcon,
  PiVerificationCheckStroke,
);
export const ShareFatIcon = makeIcon(PhShareFatIcon, PiShare01Stroke);
export const ShareIcon = makeIcon(PhShareIcon, PiShare01Stroke);
export const ShareNetworkIcon = makeIcon(PhShareNetworkIcon, PiShare02Stroke);
export const ShieldCheckIcon = makeIcon(PhShieldCheckIcon, PiShieldCheckStroke);
export const ShieldIcon = makeIcon(PhShieldIcon, PiShieldStroke);
export const SidebarSimpleIcon = makeIcon(
  PhSidebarSimpleIcon,
  PiSidebarDefaultStroke,
);
export const SignInIcon = makeIcon(PhSignInIcon, PiLogInRightStroke);
export const SignOut = makeIcon(PhSignOut, PiLogOutRightStroke);
export const SignOutIcon = makeIcon(PhSignOutIcon, PiLogOutRightStroke);
export const SlidersHorizontalIcon = makeIcon(
  PhSlidersHorizontalIcon,
  PiFilterHorizontalStroke,
);
export const SmileySadIcon = makeIcon(PhSmileySadIcon, PiFaceSadStroke);
export const SparkleIcon = makeIcon(PhSparkleIcon, PiSparkleAI01Stroke);
export const SpeakerHigh = makeIcon(PhSpeakerHigh, PiSpeakerOnStroke);
export const Spinner = makeIcon(PhSpinner, PiSpinnerStroke);
export const SpinnerGapIcon = makeIcon(PhSpinnerGapIcon, PiSpinnerStroke);
export const SpinnerIcon = makeIcon(PhSpinnerIcon, PiSpinnerStroke);
export const SquareIcon = makeIcon(PhSquareIcon, PiSquareStroke);
export const SquaresFour = makeIcon(PhSquaresFour, PiGrid01Stroke);
export const SquaresFourIcon = makeIcon(PhSquaresFourIcon, PiGrid01Stroke);
export const StackIcon = makeIcon(PhStackIcon, PiLayersToStroke);
export const Star = makeIcon(PhStar, PiStarStroke);
export const StarIcon = makeIcon(PhStarIcon, PiStarStroke);
export const Stop = makeIcon(PhStop, PiStopBigStroke);
export const StopCircleIcon = makeIcon(PhStopCircleIcon, PiStopCircleStroke);
export const StopIcon = makeIcon(PhStopIcon, PiStopSquareStroke);
export const StorefrontIcon = makeIcon(PhStorefrontIcon, PiStoreDefaultStroke);
export const SunIcon = makeIcon(PhSunIcon, PiSunStroke);
export const Table = makeIcon(PhTable, PiGridTableStroke);
export const TableIcon = makeIcon(PhTableIcon, PiGridTableStroke);
export const TerminalIcon = makeIcon(
  PhTerminalIcon,
  PiTerminalConsoleSquareStroke,
);
export const TextBIcon = makeIcon(PhTextBIcon, PiBoldStroke);
export const TextItalicIcon = makeIcon(PhTextItalicIcon, PiItalicStroke);
export const TextStrikethroughIcon = makeIcon(
  PhTextStrikethroughIcon,
  PiStrikeThroughStroke,
);
export const ThumbsDown = makeIcon(PhThumbsDown, PiThumbReactionDislikeStroke);
export const ThumbsUp = makeIcon(PhThumbsUp, PiThumbReactionLikeStroke);
export const TiktokLogo = makeIcon(PhTiktokLogo, PiTiktokStroke);
export const Trash = makeIcon(PhTrash, PiDeleteDustbin01Stroke);
export const TrashIcon = makeIcon(PhTrashIcon, PiDeleteDustbin01Stroke);
export const Tray = makeIcon(PhTray, PiInboxDefaultStroke);
export const TreeStructureIcon = makeIcon(
  PhTreeStructureIcon,
  PiGitBranchStroke,
);
export const TwitterLogoIcon = makeIcon(PhTwitterLogoIcon, PiTwitterStroke);
export const UploadIcon = makeIcon(PhUploadIcon, PiUploadUpStroke);
export const UploadSimple = makeIcon(PhUploadSimple, PiUploadUpStroke);
export const UploadSimpleIcon = makeIcon(PhUploadSimpleIcon, PiUploadUpStroke);
export const User = makeIcon(PhUser, PiUserDefaultStroke);
export const UserCheck = makeIcon(PhUserCheck, PiUserCheckStroke);
export const UserCircle = makeIcon(PhUserCircle, PiUserCircleStroke);
export const UserCircleDashedIcon = makeIcon(
  PhUserCircleDashedIcon,
  PiUserCircleDottedStroke,
);
export const UserCircleIcon = makeIcon(PhUserCircleIcon, PiUserCircleStroke);
export const UserIcon = makeIcon(PhUserIcon, PiUserDefaultStroke);
export const UserMinus = makeIcon(PhUserMinus, PiUserRemoveStroke);
export const UserPlus = makeIcon(PhUserPlus, PiUserPlusStroke);
export const Users = makeIcon(PhUsers, PiUserTwoStroke);
export const UsersIcon = makeIcon(PhUsersIcon, PiUserTwoStroke);
export const VideoCamera = makeIcon(PhVideoCamera, PiVideoRecordingStroke);
export const VideoCameraIcon = makeIcon(
  PhVideoCameraIcon,
  PiVideoRecordingStroke,
);
export const WalletIcon = makeIcon(PhWalletIcon, PiWalletDefaultStroke);
export const Warning = makeIcon(PhWarning, PiAlertTriangleStroke);
export const WarningCircle = makeIcon(PhWarningCircle, PiAlertCircleStroke);
export const WarningCircleIcon = makeIcon(
  PhWarningCircleIcon,
  PiAlertCircleStroke,
);
export const WarningDiamondIcon = makeIcon(
  PhWarningDiamondIcon,
  PiAlertTriangleStroke,
);
export const WarningIcon = makeIcon(PhWarningIcon, PiAlertTriangleStroke);
export const WarningOctagonIcon = makeIcon(
  PhWarningOctagonIcon,
  PiAlertTriangleStroke,
);
export const WebhooksLogoIcon = makeIcon(PhWebhooksLogoIcon, PiWebhookStroke);
export const X = makeIcon(PhX, PiMultipleCrossCancelDefaultStroke);
export const XCircleIcon = makeIcon(
  PhXCircleIcon,
  PiMultipleCrossCancelCircleStroke,
);
export const XIcon = makeIcon(PhXIcon, PiMultipleCrossCancelDefaultStroke);
export const XLogo = makeIcon(PhXLogo, PiXComStroke);
export const YoutubeLogo = makeIcon(PhYoutubeLogo, PiYoutubeStroke);
