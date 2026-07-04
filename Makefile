# Docker workflow for gpu-lidar-localization.
# See docker/ for the image definition and README.md for usage.

COMPOSE := docker compose -f docker/compose.yaml
SERVICE := localizer

.DEFAULT_GOAL := help

.PHONY: help build up down shell colcon clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	$(COMPOSE) build

up: ## Start the container in the background (GPU + host network + X11)
	@xhost +local:root > /dev/null 2>&1 || true
	$(COMPOSE) up -d

down: ## Stop and remove the container
	$(COMPOSE) down

shell: up ## Open an interactive shell inside the running container
	$(COMPOSE) exec $(SERVICE) bash

colcon: up ## Build the ROS 2 workspace inside the container
	$(COMPOSE) exec $(SERVICE) bash -c \
		'source /opt/ros/humble/setup.bash && cd /ws/cuda_icp_localizer && colcon build --symlink-install'

clean: ## Remove the container together with its build volumes
	$(COMPOSE) down --volumes
