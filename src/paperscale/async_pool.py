"""Async concurrency primitives for the full-lifecycle worker pool.

``AdaptiveLimiter`` is a resizable concurrency gate. A plain ``asyncio.Semaphore``
cannot change its bound at runtime, but AIMD needs exactly that: additive-increase
on sustained success up to a ceiling, multiplicative-decrease toward a floor on
overload. The limiter caps concurrent holders at a *mutable* ``limit`` in
``[1, maximum]`` and wakes waiters when the limit grows.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AdaptiveLimiter:
    def __init__(self, limit: int, *, maximum: int) -> None:
        self._maximum = max(1, maximum)
        self._limit = max(1, min(limit, self._maximum))
        self._in_use = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def maximum(self) -> int:
        return self._maximum

    async def set_limit(self, value: int) -> None:
        async with self._cond:
            new_limit = max(1, min(int(value), self._maximum))
            grew = new_limit > self._limit
            self._limit = new_limit
            if grew:
                # Wake enough waiters to fill the newly available slots.
                self._cond.notify_all()

    async def acquire(self) -> None:
        async with self._cond:
            while self._in_use >= self._limit:
                await self._cond.wait()
            self._in_use += 1

    async def release(self) -> None:
        async with self._cond:
            if self._in_use > 0:
                self._in_use -= 1
            self._cond.notify(1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()
