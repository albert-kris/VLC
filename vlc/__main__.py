"""使 `python -m vlc` 走包根 main.py。"""

from vlc.main import main

if __name__ == "__main__":
    raise SystemExit(main())
