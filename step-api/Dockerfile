# Base image with build tools and Python
FROM ubuntu:22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js 16, build tools, and Python 3
RUN apt-get update && apt-get install -y \
  curl \
  build-essential \
  cmake \
  python3 \
  python3-pip \
  python3-distutils \
  git \
  pkg-config \
  && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
  && apt-get install -y nodejs

# Install node-gyp globally
RUN npm install -g node-gyp

# Create app directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy remaining files
COPY . .

# Expose port
EXPOSE 3000

# Start the server
CMD [ "npm", "start" ]