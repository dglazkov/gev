FROM node:24-slim
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY src ./src
COPY public ./public
USER node
# Node 24 runs TypeScript directly (type stripping), so there is no build step.
CMD ["node", "src/server.ts"]
