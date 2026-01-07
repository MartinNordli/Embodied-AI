.PHONY: help
help:
	@echo "Targets:"
	@echo "  lint        - run ruff"
	@echo "  test        - run pytest"
	@echo "  tree        - show repo structure (excluding ignored dirs)"

.PHONY: lint
lint:
	python -m ruff check .

.PHONY: test
test:
	python -m pytest -q

.PHONY: tree
tree:
	@find . -maxdepth 3 -type d \\( -name .git -o -name .venv -o -name runs -o -name data_local \\) -prune -false -o -print


