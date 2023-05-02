# Use the official Node.js image as the base image
FROM node:19-alpine

ARG NODE_ENV

ENV NODE_ENV=$NODE_ENV

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application code
COPY . .
RUN mv .env.docker .env  \
    && sed -ie 's/postgresql/sqlite/g' prisma/schema.prisma \
    && sed -ie 's/mysql/sqlite/g' prisma/schema.prisma \
   && sed -ie 's/@db.Text//' prisma/schema.prisma

# Expose the port the app will run on
EXPOSE 3000

# Add Prisma and generate Prisma client
RUN npx prisma generate  \
    && npx prisma migrate dev --name init  \
    && npx prisma db push

# Build the Next.js app
RUN npm run build


# Start the application
CMD ["npm", "start"]
