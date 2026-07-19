import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import LayoutShell from "@/layout/LayoutShell";
import { AuthProvider } from "@/hooks/useAuth";
import "@/app/globals.css";


const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "E-Library - Discover Your Next Favorite Book",
  description: "Your ultimate platform for discovering, exploring, and connecting with books and fellow readers worldwide. Access thousands of books across all genres.",
  keywords: ["books", "library", "reading", "recommendations", "ebook"],
  authors: [{ name: "E-Library Team" }],
  icons: {
    icon: "next.svg",
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          <LayoutShell>{children}</LayoutShell>
        </AuthProvider>
      </body>
    </html>
  );
}
