# StopSign AI Development Makefile
# Simplifies local development while keeping production unchanged

.PHONY: help dev dev-build dev-down dev-logs dev-clean setup test prod-help

# Default target
help:
	@echo "🚦 StopSign AI Development Commands"
	@echo ""
	@echo "Local Development:"
	@echo "  make setup      - Initial setup for local development"
	@echo "  make dev        - Start local development stack"
	@echo "  make dev-build  - Rebuild and start local development"
	@echo "  make dev-down   - Stop local development stack"
	@echo "  make dev-logs   - Follow logs from local stack"
	@echo "  make dev-clean  - Clean local volumes and containers"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run test suite (when available)"
	@echo ""
	@echo "Production:"
	@echo "  make prod-help  - Show production deployment info"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Clean up build artifacts"

# Local Development Commands
setup:
	@echo "🚀 Setting up local development environment..."
	@echo "🔍 Checking setup requirements..."
	@mkdir -p volumes/{redis,postgres,minio,hls}
	@echo "📁 Created volume directories (safe to run multiple times)"
	@if [ ! -f docker/local/.env ]; then \
		echo "📋 Creating docker/local/.env from template..."; \
		cp docker/local/.env.example docker/local/.env; \
		echo "✅ Created docker/local/.env - you can customize this file"; \
	else \
		echo "✅ docker/local/.env already exists (no changes made - your settings are safe)"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		if [ ! -f sample_data/sample.mp4 ]; then \
			echo "🎥 Creating sample video with ffmpeg..."; \
			ffmpeg -f lavfi -i "testsrc=duration=30:size=640x480:rate=15" \
			       -f lavfi -i "sine=frequency=1000:duration=30" \
			       -c:v libx264 -crf 28 -c:a aac \
			       sample_data/sample.mp4 -y -loglevel quiet; \
			echo "✅ Sample video created at sample_data/sample.mp4"; \
		else \
			echo "📹 Sample video already exists (no changes made)"; \
		fi; \
	else \
		echo "⚠️  ffmpeg not found - you'll need to provide sample_data/sample.mp4 manually"; \
	fi
	@echo ""
	@echo "🎉 Setup complete! This is safe to run multiple times."
	@echo "📝 Next steps:"
	@echo "   1. Customize docker/local/.env if needed"
	@echo "   2. Run 'make dev' to start the development stack"

dev:
	@echo "🚀 Starting local development stack (CPU-only, lightweight)..."
	cd docker/local && docker compose up

dev-build:
	@echo "🔨 Rebuilding local development stack (CPU-only, ~200MB vs 3GB prod)..."
	cd docker/local && docker compose up --build

dev-down:
	@echo "🛑 Stopping local development stack..."
	cd docker/local && docker compose down

dev-logs:
	@echo "📋 Following logs from local development stack..."
	cd docker/local && docker compose logs -f

dev-clean:
	@echo "🧹 Cleaning local development environment..."
	cd docker/local && docker compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete"

# Production Info (single release stack)
prod-help:
	@echo "🏭 Production Deployment"
	@echo ""
	@echo "The manual-app production release uses one Compose file for all pipeline services:"
	@echo "  docker/production/docker-compose.yml - RTSP, analyzer, ffmpeg, and web"
	@echo "  /home/drose/manual-apps/stopsign_ai/rtsp_to_redis/.env - cube-only RTSP env authority"
	@echo ""
	@echo "Run manual-app deploy stopsign to build/recreate every service from one checkout."
	@echo "Runtime secrets remain on cube and are never committed."

# Testing
# The suite spans the analyzer (ml), the database and storage layers, the web app,
# and Lookout — so the test environment needs those extras, not just pytest.
TEST_EXTRAS = --extra ml --extra db --extra storage --extra web --extra lookout --extra test

test:
	@echo "🧪 Running tests..."
	uv run $(TEST_EXTRAS) pytest tests/ -v

test-lookout:
	@echo "🧪 Running the Lookout suite only (fast, offline)..."
	uv run --extra lookout --extra test pytest tests/test_lookout.py -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	uv run $(TEST_EXTRAS) pytest tests/ -v --cov=stopsign --cov-report=term-missing

test-watch:
	@echo "🧪 Running tests in watch mode..."
	uv run $(TEST_EXTRAS) pytest tests/ -v --tb=short -x

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "✅ Clean complete"