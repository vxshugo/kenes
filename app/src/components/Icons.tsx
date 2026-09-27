import type { ReactNode } from "react";
import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...rest }: P & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconRecord = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="6" fill="currentColor" stroke="none" />
  </Svg>
);
export const IconStop = (p: P) => (
  <Svg {...p}>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor" stroke="none" />
  </Svg>
);
export const IconCopy = (p: P) => (
  <Svg {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V6a2 2 0 0 1 2-2h8" />
  </Svg>
);
export const IconCheck = (p: P) => (
  <Svg {...p}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Svg>
);
export const IconDownload = (p: P) => (
  <Svg {...p}>
    <path d="M12 4v11M7 10.5l5 5 5-5M5 20h14" />
  </Svg>
);
export const IconTrash = (p: P) => (
  <Svg {...p}>
    <path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 12.5h9l1-12.5" />
  </Svg>
);
export const IconRefresh = (p: P) => (
  <Svg {...p}>
    <path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4" />
  </Svg>
);
export const IconChevron = (p: P) => (
  <Svg {...p}>
    <path d="M9 6l6 6-6 6" />
  </Svg>
);
export const IconSend = (p: P) => (
  <Svg {...p}>
    <path d="M4.5 12h14M13 6.5l5.5 5.5-5.5 5.5" />
  </Svg>
);
export const IconSpark = (p: P) => (
  <Svg {...p}>
    <path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z" />
    <path d="M18.5 16.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z" />
  </Svg>
);
export const IconTranslate = (p: P) => (
  <Svg {...p}>
    <path d="M4 5.5h9M8.5 4v1.5M6 5.5c.8 3 3 5.5 6 7M11 5.5c-.8 3.5-3 6.5-6.5 8" />
    <path d="M13 20l3.5-8 3.5 8M14.2 17.3h4.6" />
  </Svg>
);
export const IconBook = (p: P) => (
  <Svg {...p}>
    <path d="M5 4.5h9.5a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3z" />
    <path d="M5 17a3 3 0 0 1 3-3h9.5" />
  </Svg>
);
export const IconClock = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8" />
    <path d="M12 8v4.5l3 1.8" />
  </Svg>
);
export const IconX = (p: P) => (
  <Svg {...p}>
    <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />
  </Svg>
);
export const IconArrowDown = (p: P) => (
  <Svg {...p}>
    <path d="M12 5v14M6.5 13.5L12 19l5.5-5.5" />
  </Svg>
);
export const IconArrowLeft = (p: P) => (
  <Svg {...p}>
    <path d="M19 12H5M10.5 6.5L5 12l5.5 5.5" />
  </Svg>
);
export const IconFile = (p: P) => (
  <Svg {...p}>
    <path d="M7 3.5h7l4.5 4.5v12.5H7z" />
    <path d="M14 3.5V8h4.5" />
  </Svg>
);
export const IconEye = (p: P) => (
  <Svg {...p}>
    <path d="M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12z" />
    <circle cx="12" cy="12" r="2.8" />
  </Svg>
);
export const IconEyeOff = (p: P) => (
  <Svg {...p}>
    <path d="M4 4l16 16M9.9 5.8A9.7 9.7 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-3 3.8M6.3 7.4A16.6 16.6 0 0 0 2.5 12s3.5 6.5 9.5 6.5a9.4 9.4 0 0 0 4.3-1" />
    <path d="M10 10.1a2.8 2.8 0 0 0 3.9 3.9" />
  </Svg>
);
export const IconUsers = (p: P) => (
  <Svg {...p}>
    <circle cx="9" cy="8.5" r="3.2" />
    <path d="M3.5 19c.6-3.2 2.8-5 5.5-5s4.9 1.8 5.5 5" />
    <path d="M15.5 5.6a3 3 0 0 1 0 5.8M17.5 14.3c1.6.7 2.7 2.2 3 4.7" />
  </Svg>
);
export const IconWand = (p: P) => (
  <Svg {...p}>
    <path d="M5 19L15.5 8.5M13.5 6.5l4 4" />
    <path d="M18.5 3.5l.6 1.5 1.5.6-1.5.6-.6 1.5-.6-1.5-1.5-.6 1.5-.6zM7.5 4l.5 1.2 1.2.5-1.2.5-.5 1.2-.5-1.2-1.2-.5 1.2-.5z" />
  </Svg>
);
export const IconMic = (p: P) => (
  <Svg {...p}>
    <rect x="9" y="3.5" width="6" height="11" rx="3" />
    <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3" />
  </Svg>
);
