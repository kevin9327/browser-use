#!/usr/bin/env python3
"""Browser-only smoke: proves BrowserSession starts (no LLM needed)."""

from __future__ import annotations

import asyncio
import sys

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession


async def _main() -> int:
	browser = BrowserSession(browser_profile=BrowserProfile(headless=True, user_data_dir=None))
	try:
		await browser.start()
		print('OK: BrowserSession started')
		return 0
	finally:
		await browser.kill()


if __name__ == '__main__':
	raise SystemExit(asyncio.run(_main()))
