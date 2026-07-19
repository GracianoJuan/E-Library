"use client";

import { usePathname } from "next/navigation";
import MainLayout from "@/layout/MainLayout";

const NO_MAIN_LAYOUT_PATHS = new Set(["/login", "/register"]);

export default function LayoutShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  if (NO_MAIN_LAYOUT_PATHS.has(pathname)) {
    return <>{children}</>;
  }

  return <MainLayout>{children}</MainLayout>;
}
