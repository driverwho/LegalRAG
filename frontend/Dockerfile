# ============================================================
# RAG Frontend - Dockerfile
# ============================================================
# Two-stage build: Node builds Vue app, Nginx serves static files

# ---- Build Stage ----
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY package.json package-lock.json* ./
RUN npm ci --no-audit

# Copy source and build
COPY . .
RUN npm run build


# ---- Production Stage ----
FROM nginx:1.27-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built static files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

# Nginx runs in foreground
CMD ["nginx", "-g", "daemon off;"]
