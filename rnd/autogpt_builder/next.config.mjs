import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'export',
    env: {
        AGPT_SERVER_URL: process.env.AGPT_SERVER_URL,
    },
    async redirects() {
        return [
            {
                source: '/',
                destination: '/build',
                permanent: false,
            },
        ];
    },
    eslint: {
        ignoreDuringBuilds: true,
    },
    typescript: {
        ignoreBuildErrors: true,
    },
};

export default nextConfig;
