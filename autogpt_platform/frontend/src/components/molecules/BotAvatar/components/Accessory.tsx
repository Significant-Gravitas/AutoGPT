import { INK, type AccessoryId, type ShapeAnchors } from "../helpers";

interface Props {
  accessory: AccessoryId;
  anchors: ShapeAnchors;
  deep: string;
}

const STAR =
  "M0,-5.5 L1.6,-1.7 L5.6,-1.6 L2.4,0.9 L3.5,4.8 L0,2.5 L-3.5,4.8 L-2.4,0.9 L-5.6,-1.6 L-1.6,-1.7 Z";

export function Accessory({ accessory, anchors, deep }: Props) {
  const { cx, eyeY, eyeGap, top, bottom, width } = anchors;
  const left = cx - width / 2;
  const right = cx + width / 2;

  switch (accessory) {
    case "glasses":
      return (
        <g fill="none" stroke={deep} strokeWidth={2.4} strokeLinecap="round">
          <circle cx={cx - eyeGap / 2} cy={eyeY} r={9.5} />
          <circle cx={cx + eyeGap / 2} cy={eyeY} r={9.5} />
          <path
            d={`M${cx - eyeGap / 2 + 9.5},${eyeY} L${cx + eyeGap / 2 - 9.5},${eyeY}`}
          />
          <path
            d={`M${cx - eyeGap / 2 - 9.5},${eyeY - 1} L${left + 2},${eyeY - 4}`}
          />
          <path
            d={`M${cx + eyeGap / 2 + 9.5},${eyeY - 1} L${right - 2},${eyeY - 4}`}
          />
        </g>
      );
    case "headset":
      return (
        <g>
          <path
            d={`M${left + 4},${eyeY} C${left + 4},${top - 6} ${right - 4},${top - 6} ${right - 4},${eyeY}`}
            fill="none"
            stroke={deep}
            strokeWidth={3.2}
            strokeLinecap="round"
          />
          <rect
            x={left - 4}
            y={eyeY - 7}
            width={9}
            height={16}
            rx={4}
            fill={deep}
            stroke={INK}
            strokeWidth={2}
          />
          <rect
            x={right - 5}
            y={eyeY - 7}
            width={9}
            height={16}
            rx={4}
            fill={deep}
            stroke={INK}
            strokeWidth={2}
          />
          <path
            d={`M${right - 1},${eyeY + 9} C${right - 1},${eyeY + 20} ${cx + 8},${eyeY + 20} ${cx + 6},${eyeY + 17}`}
            fill="none"
            stroke={deep}
            strokeWidth={2.4}
            strokeLinecap="round"
          />
          <circle cx={cx + 6} cy={eyeY + 17} r={2.2} fill={deep} />
        </g>
      );
    case "cap":
      return (
        <g>
          <path
            d={`M${cx - 26},${top + 10} A26,24 0 0 1 ${cx + 26},${top + 10} L${cx + 26},${top + 13} L${cx - 26},${top + 13} Z`}
            fill={deep}
            stroke={INK}
            strokeWidth={2}
            strokeLinejoin="round"
          />
          <rect
            x={cx - 2}
            y={top + 9}
            width={44}
            height={6}
            rx={3}
            fill={deep}
            stroke={INK}
            strokeWidth={2}
          />
          <circle
            cx={cx}
            cy={top - 12}
            r={2.4}
            fill={deep}
            stroke={INK}
            strokeWidth={1.5}
          />
        </g>
      );
    case "pen":
      return (
        <g transform={`translate(${right - 12} ${top + 2}) rotate(28)`}>
          <rect
            x={-3}
            y={-30}
            width={6}
            height={30}
            rx={1.5}
            fill={deep}
            stroke={INK}
            strokeWidth={1.8}
          />
          <path
            d="M-3,0 L0,7 L3,0 Z"
            fill="#F4E3B4"
            stroke={INK}
            strokeWidth={1.8}
            strokeLinejoin="round"
          />
          <rect
            x={-3}
            y={-30}
            width={6}
            height={5}
            rx={1}
            fill="#fff"
            stroke={INK}
            strokeWidth={1.8}
          />
        </g>
      );
    case "star":
      return (
        <g transform={`translate(${right - 9} ${bottom - 12})`}>
          <circle r={9} fill={deep} stroke={INK} strokeWidth={2} />
          <path d={STAR} fill="#fff" />
        </g>
      );
    case "bow":
      return (
        <g transform={`translate(${left + 12} ${top + 8}) rotate(-18)`}>
          <path
            d="M0,0 L-11,-6 L-11,6 Z"
            fill={deep}
            stroke={INK}
            strokeWidth={1.8}
            strokeLinejoin="round"
          />
          <path
            d="M0,0 L11,-6 L11,6 Z"
            fill={deep}
            stroke={INK}
            strokeWidth={1.8}
            strokeLinejoin="round"
          />
          <circle r={2.6} fill={deep} stroke={INK} strokeWidth={1.8} />
        </g>
      );
    case "badge":
      return (
        <g transform={`translate(${left + 8} ${bottom - 28})`}>
          <rect
            width={18}
            height={12}
            rx={2.5}
            fill="#fff"
            stroke={deep}
            strokeWidth={2}
          />
          <circle cx={5} cy={6} r={2} fill={deep} />
          <path
            d="M9,4.5 L14.5,4.5 M9,7.5 L13,7.5"
            stroke={deep}
            strokeWidth={1.6}
            strokeLinecap="round"
          />
        </g>
      );
    default:
      return null;
  }
}
