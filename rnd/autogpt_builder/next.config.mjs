/** @type {import('next').NextConfig} */
const nextConfig = {
    async redirects() {
        return [
            {
                source: '/',
                destination: '/build',
                permanent: false,
            },
        ]
    }
};

export default nextConfig;
