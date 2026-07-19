import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "covers.openlibrary.org",
      },
      {
        protocol: "https",
        hostname: "images-na.ssl-images-amazon.com",
      },
      {
        protocol: "https",
        hostname: "*.ssl-images-amazon.com",
      },
      {
        protocol: 'https',
        hostname: 'image.gramedia.net',
      },
    ],
  },
};

export default nextConfig;
