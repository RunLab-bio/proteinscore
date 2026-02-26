"""
ProteinScore Structured Logging

Structlog configuration for consistent, structured logging across the application.
Follows Python Backend Best Practices (PBP) Section 2.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from proteinscore.config import get_default_config


def configure_logging() -> None:
    """Configure structlog with appropriate processors for environment."""
    config = get_default_config()

    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.UnicodeDecoder(),
    ]

    # Production: JSON output for log aggregation
    # Development: Colorized console output
    if config.log_level == "DEBUG":
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level),
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a logger instance with optional initial context.

    Args:
        name: Logger name (usually __name__)
        **initial_context: Initial context key-value pairs

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("scoring_started", sequence_length=150)
        >>> logger.error("prediction_failed", predictor="stability", error=str(exc))
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """
    Bind context variables to all loggers in the current context.

    Useful for adding request-scoped context like request_id, user_id, etc.

    Args:
        **context: Key-value pairs to bind

    Example:
        >>> bind_context(request_id="abc-123", user_id="user-456")
        >>> logger.info("processing")  # Will include request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# Module-level logger for imports
logger = get_logger("proteinscore")


__all__ = [
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
    "logger",
]
