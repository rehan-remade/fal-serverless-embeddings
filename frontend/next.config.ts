import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['@lancedb/lancedb']
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Mark native modules as external on server-side
      config.externals.push('@lancedb/lancedb');
    }
    return config;
  }
};

export default nextConfig;
