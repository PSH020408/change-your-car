import type { Metadata } from "next";
import { Barlow, IBM_Plex_Mono } from "next/font/google";
import "@/styles/globals.css";

const barlow = Barlow({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-barlow" });
const plex = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-plex-mono" });

export const metadata: Metadata = {
  title: "F1 Virtual Sim",
  description: "Setup and conditions simulator on real ground-effect-era telemetry",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${barlow.variable} ${plex.variable}`}>
      <body>{children}</body>
    </html>
  );
}
