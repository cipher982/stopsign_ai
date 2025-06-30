# StopSign AI Local Development
# Simple commands to manage local vs production viewing

.PHONY: help local prod-view stop logs

help:	## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

local:	## Start local development (empty data, sample video)
	@echo "🏠 Starting local development at http://localhost:8000"
	@echo "📹 Using sample video and empty local database"
	@echo "🛑 Use Ctrl+C to stop"
	docker compose up

prod-view:	## Start production view mode (real data, real camera)
	@echo "🌐 Starting production view mode at http://localhost:8000"
	@echo "📊 Connected to production database (31,799+ vehicle passes)"
	@echo "📹 Connected to production RTSP camera"
	@echo "🔒 READ-ONLY: No data will be written to production"
	@echo "🛑 Use Ctrl+C to stop"
	docker compose -f docker-compose.yml -f docker-compose.prod-view.yml up

stop:	## Stop all services
	docker compose down
	@echo "🛑 All services stopped"

logs:	## Show logs (use 'make logs SERVICE=web_server' for specific service)
	@if [ -n "$(SERVICE)" ]; then \
		docker compose logs -f $(SERVICE); \
	else \
		docker compose logs -f; \
	fi

clean:	## Clean up Docker containers and volumes
	docker compose down -v
	docker system prune -f
	@echo "🧹 Cleaned up containers and volumes"