import type { NextConfig } from "next";

/**
 * The HUD is entirely client-rendered, so it ships as static files. In
 * production FastAPI serves this `out/` directory on the same origin as the
 * API (Dockerfile), which is why NEXT_PUBLIC_API_BASE_URL is "" there.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
