"""Shared helpers: logging, rate limiting, caching, and simple text utilities."""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

#Generate type variable used by functions that should preserve the input/output type.
T =TypeVar("T")

# Regular expressions used to identify Claim_IDs and Policy IDs in text.

CLAIM_ID_PATTERN = re.compile(r"\bCLM-\d{4}\b", re.IGNORECASE) #CLM-1023
POLICY_ID_PATTERN = re.compile(r"\bPOL-\d{4}\b", re.IGNORECASE) #POL-4582

def setup_logging(level: int = logging.INFO) -> logging.Logger:
  """
  configure and return the application logger.
  
  The logger prints messages to the console with a timestamp,
  log level, and message.
  
  Example:
      logger = setup_logging()
      logger.info("Application started")
  Output:
  2026-08-22 10:15:30 |INFO | Application started
  """
  # Get the application-specific logger
  # Calling getLogger() with the same name returns the same logger.

  logger = logging.getLogger("claims_assistant")

  # Add a console handler only if one doesn't already exist.
  # This prevents duplicate log message log messages if setup_logging() is called multiple times.

  if not logger.handlers:
    handler = logging.StreamHandler()

    # Define how each log message should look:
    # timestamp | log level |message

    handler.setFormatter(
      logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
  # Set the minimum level of messages that should be recorded.
  logger.setLevel(level)

  return logger
logger = setup_logging()


class RateLimiter:
  """
  Simple in-memory rate limiter using a sliding time window.
  
  It limits how many times an action can be performed within a given time period

  Examples:
      RateLimiter(max_calls=5, period_seconds=60)

      Allows 5 calls within 60 seconds.
      the 6th call is rejected until an earlier call expires.

  """
  def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
    self.max_calls = max_calls
    self.period_seconds = period_seconds

    # Store the timestamp of each recent call.
    self._timestamps: deque[float] = deque()
  def allow(self) -> bool:
    """
    Check whether a new call is allowed.
    
    Removes expired timestamp first, then checks whether
    the current number of calls gas reached the limit.
    
    Returns:
      True if the call is allowed, otherwise False.
    """

    now = time.monotonic()

    # Remove calls that are outside the current time window.
    while self._timestamps and now - self._timestamps[0] > self.period_seconds:
      self._timestamps.popleft()

    # Reject the call if the limit has already been reached.
    if len(self._timestamps) >= self.max_calls:
      return False

    # Record this call and allow it
    self._timestamps.append(now)
    return True

  def remaining(self) -> int:
    """
    Return the number of calls available in the current window.
    Example:
    
      Limit = 5, calls already made = 2,
      Returns: 3
    """

    now = time.monotonic()

    #Remove expired calls before calculating the remaining limit.
    while self._timestamps and now - self._timestamps[0] > self.period_seconds:
      self._timestamps.popleft()
    return max(0, self.max_calls - len(self._timestamps))


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """
    Execute a function and retry it when configured exceptions occur.

    max_retries represents the total number of attempts,
    including the initial call.

    Delay increases exponentially:
    1s -> 2s -> 4s -> 8s.

    Only the specified exception types are retried.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    if base_delay < 0:
        raise ValueError("base_delay cannot be negative")

    for attempt in range(max_retries):
        try:
            return fn()

        except exceptions as exc:

            # Final attempt: propagate the original exception.
            if attempt == max_retries - 1:
                logger.error(
                    "All %s attempts failed: %s",
                    max_retries,
                    exc,
                )
                raise

            delay = base_delay * (2 ** attempt)

            logger.warning(
                "Retry %s/%s after error: %s (sleep %.1fs)",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )

            time.sleep(delay)


def truncate_context(texts: list[str], max_chars: int) -> list[str]:
  """
  select documents until the total character limit is reached
  
  if the final document exceeds the remaining space , only the 
  required portion is kept and marked as truncated.
  
  Example:
    max_chars = 100,
    Documents are added until the combined text reaches 1000 characters.
  """

  selected: list[str] =[]
  total = 0

  for text in texts:
    # Stop once the context budget has been reached.
    if total >= max_chars:
      break

    remaining = max_chars - total

    if len(text) <= remaining:
      selected.append(text)
      total += len(text)
    else:
      #keep only the portion that fits within the remaining budget.

      selected.append(text[:remaining] + "\n...[truncated]")
      total = max_chars
  return selected

def extract_claim_ids(text: str) -> list[str]:
  """
  Extract all Claim ids from a text string.
  
  Example:
      "Review CLM-1023 and CLM-1045"
      >["CLM-1023", "CLM-1045"]
      
  """

  # Find all matching policy ids and normalize them to uppercase
  return [m.group(0).upper() for m in CLAIM_ID_PATTERN.finditer(text)]

def extract_policy_ids(text: str) -> list[str]:
  """
  Extract all policy id from a text string
  
  Example:
     "check POL-1001 and POL-1002"
     >["POL-1001", "POL-1002"]
  """

  # Find all matching claim ids and normalize them to uppercase.
  return [m.group(0).upper() for m in POLICY_ID_PATTERN.finditer(text)]

def tokenize(text: str) -> list[str]:
  """
  Convert text into simple normalized tokens for BM25 search.
  
  The function converts text to lowercase, extract words/numbers
  and removes a trailing "s" from simple plural words.
  
  Example:
      "Accidents involve damaged vehicles"
      >["Accident", "involve", "damaged", "vehicle"]
  """

  # Extract lowercase words, numbers and dollar values.
  tokens = re.findall(r"[a-z0-9$]+", text.lower())

  normalized: list[str] = []

  for token in tokens:
    # Convert simple plurals to singular form so that
    # "accidents" can match "accident".

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
      normalized.append(token[:-1])
    else:
      normalized.append(token)
  return normalized


def format_money(value: float) -> str:
  """
  Format a number as a US currency string.
  
  Example:
      12500.5 > "$12,500.50"
  """

  return f"${value:,.2f}"

def safe_get(metadata: dict[str, Any], key: str, default: str ="") -> str:
  """
  Safely retrieve a value from a metadata dictionary.
  
  returns the default value if the key is missing, and converts
  non-string values to strings.
  
  Example:
   safe_get({"claim_id": "CLM-1000"}, "claim_id")
   > "CLM-1001"
   
   safe_get({}, "claim_id")
  """

  value =  metadata.get(key, default)
  # Handle explicit NOne values gracefully.
  return "" if value is None else str(value)





  
    
    
