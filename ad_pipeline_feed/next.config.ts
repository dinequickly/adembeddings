import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'nlobsjnpcjxfabhnbvza.supabase.co',
        pathname: '/storage/v1/object/public/ads/**',
      },
    ],
  },
};

export default nextConfig;
