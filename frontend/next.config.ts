import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // the HUD talks to FastAPI directly (NEXT_PUBLIC_API_BASE_URL); nothing server-side here
  output: "standalone",
};

export default nextConfig;
